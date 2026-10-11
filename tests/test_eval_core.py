"""Evaluation plumbing (Phase 4): bootstrap, windows, masks, per-window loop, summary."""

import numpy as np
import pytest
import torch

from evflow.eval.bootstrap import block_length, moving_block_ci
from evflow.eval.run import CMPredictor, DatasetAccess, ZeroPredictor, evaluate_windows
from evflow.eval.summary import summarise
from evflow.eval.windows import Window, dsec_windows, events_in_window, mvsec_windows, region_mask
from evflow.models.cm_baseline import CMParams


def test_block_length_from_stride():
    t = np.arange(0, 10_000_000, 22_000)  # 22 ms stride
    assert block_length(t, 1.0) == 45
    assert block_length(np.array([5]), 1.0) == 1


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    v = 2.0 + np.convolve(rng.normal(size=600), np.ones(20) / 20, mode="same")
    lo, hi = moving_block_ci(v, block=30, seed=1)
    assert lo < v.mean() < hi
    assert (lo, hi) == moving_block_ci(v, block=30, seed=1)


def test_bootstrap_wider_with_blocks_on_correlated_series():
    rng = np.random.default_rng(1)
    v = np.repeat(rng.normal(size=40), 25)  # strongly autocorrelated
    lo1, hi1 = moving_block_ci(v, block=1, seed=0)
    lo25, hi25 = moving_block_ci(v, block=25, seed=0)
    assert (hi25 - lo25) > 2 * (hi1 - lo1)


def test_bootstrap_constant_and_errors():
    assert moving_block_ci(np.full(10, 3.0), block=4) == (3.0, 3.0)
    with pytest.raises(ValueError):
        moving_block_ci(np.array([1.0, np.nan]), block=1)
    with pytest.raises(ValueError):
        moving_block_ci(np.array([]), block=1)


def test_mvsec_windows_and_predecessors():
    ts = np.arange(100) * 1000 + 5
    w = mvsec_windows(ts, aps_first=10, aps_last=20, dt=4)
    assert len(w) == 11
    assert (w[0].t_start_us, w[0].t_end_us, w[0].prev_start_us) == (10_005, 14_005, 6_005)
    with pytest.raises(ValueError):
        mvsec_windows(ts, aps_first=10, aps_last=96, dt=4)


def test_dsec_windows_contiguity():
    ts = np.array([[0, 100_000], [100_000, 200_010], [300_000, 399_990]])
    w = dsec_windows(ts, delta_t_us=100_000)
    assert [x.prev_start_us for x in w] == [None, 0, None]
    with pytest.raises(ValueError):
        dsec_windows(np.array([[0, 50_000]]), delta_t_us=100_000)


def test_region_and_events_masks():
    m = region_mask({"rows": [2, 258], "cols": [45, 301]}, 260, 346)
    assert m.sum() == 256 * 256 and m[2, 45] and not m[258, 45] and not m[2, 301]
    e = events_in_window(np.array([0, 5, 400]), np.array([0, 3, 1]), 10, 10)
    assert e.sum() == 2 and e[3, 5]
    with pytest.raises(ValueError):
        events_in_window(np.array([0.5]), np.array([1.0]), 10, 10)


class _Slicer:
    def __init__(self, x, y, t, p):
        self.ev = dict(x=x, y=y, t=t, p=p)

    def get_events(self, a, b):
        m = (self.ev["t"] >= a) & (self.ev["t"] < b)
        return {k: v[m] for k, v in self.ev.items()}


def _toy():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 40, n).astype(np.uint16)
    y = rng.integers(0, 30, n).astype(np.uint16)
    t = np.sort(rng.integers(0, 100_000, n)).astype(np.int64)
    p = rng.integers(0, 2, n).astype(np.int8)
    gt = np.zeros((2, 30, 40))
    gt[0] = 1.5
    gt[1] = -2.0
    data = DatasetAccess(slicer=_Slicer(x, y, t, p),
                         gt=lambda w: (gt * (w.t_end_us - w.t_start_us) / 10_000,
                                       np.ones((30, 40), bool)),
                         height=30, width=40, region={"rows": [0, 30], "cols": [0, 40]},
                         sparse=True)
    wins = [Window(k, k * 10_000, (k + 1) * 10_000, (k - 1) * 10_000 if k else None)
            for k in range(10)]
    return data, wins


def test_zero_predictor_epe_is_mean_gt_magnitude_and_fwl_one():
    data, wins = _toy()
    pw = evaluate_windows(wins, data, ZeroPredictor(30, 40))
    assert np.allclose(pw["epe"], 2.5)
    assert np.allclose(pw["fwl"], 1.0)
    s = summarise(pw, block_seconds=0.03, resamples=200, ci=0.95, seed=0)
    assert s["metrics"]["epe"] == pytest.approx(2.5)
    assert s["n_windows"] == 10 and s["bootstrap"]["block_windows"] == 3
    assert s["ci95"]["epe"] == pytest.approx([2.5, 2.5])


def test_cm_predictor_runs_and_summary_keys():
    data, wins = _toy()
    pred = CMPredictor(CMParams(tile_px=16, iters_global=6, iters_tiles=6), 30, 40, "cpu")
    pw = evaluate_windows(wins[:3], data, pred)
    assert pw["epe"].shape == (3,) and np.all(np.isfinite(pw["epe"]))
    s = summarise(pw, block_seconds=1.0, resamples=50, ci=0.95, seed=0)
    assert set(s["metrics"]) == {"epe", "outlier_pct", "npe_1", "npe_2", "npe_3", "ae_deg", "fwl"}


def test_skipped_windows_are_counted():
    data, wins = _toy()
    data.gt = lambda w: (np.zeros((2, 30, 40)), np.zeros((30, 40), bool))
    pw = evaluate_windows(wins[:4], data, ZeroPredictor(30, 40))
    s = summarise(pw, block_seconds=1.0, resamples=10, ci=0.95, seed=0)
    assert s["n_windows"] == 0 and s["n_windows_skipped_no_valid_px"] == 4
    assert s["metrics"]["epe"] is None
    assert torch.isfinite(torch.tensor(s["metrics"]["fwl"]))


class _FakeSlicer:
    def __init__(self, name, t_lo, t_hi):
        self.name, self.t_lo, self.t_hi = name, t_lo, t_hi

    def get_events(self, a, b):
        return {"src": self.name} if self.t_lo <= a and b <= self.t_hi else None


def test_chunked_slicer_routes_by_owned_start():
    from evflow.eval.datasets import ChunkedSlicer
    # chunk p0 owns starts 100..500 and covers events up to 900 (dt tail); p1 owns 600..
    s = ChunkedSlicer([(600, 1500, _FakeSlicer("p1", 400, 2000)),
                       (100, 500, _FakeSlicer("p0", 0, 900))])
    assert s.get_events(500, 880)["src"] == "p0"   # owned by p0 although p1 also covers it
    assert s.get_events(600, 700)["src"] == "p1"
    assert s.get_events(500, 950) is None           # owner does not cover the end -> uncovered
    with pytest.raises(ValueError):
        s.get_events(550, 560)                      # owned by no chunk
    with pytest.raises(ValueError):
        ChunkedSlicer([(0, 10, None), (10, 20, None)])
