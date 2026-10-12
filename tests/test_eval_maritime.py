"""T3 evaluation core (Phase 7 step 4; D-057) on synthetic windows: teacher-EPE family under the
FB / events / class masks equals the numpy reference, FWL/RSAT per slice, cluster bootstrap."""

import numpy as np
import pytest
import torch
from test_cm_baseline import moving_dots

from evflow.eval import maritime as em
from evflow.eval.run import ZeroPredictor
from evflow.eval.windows import Window
from evflow.flow import metrics_ref
from evflow.teacher.cache import write
from evflow.train.data import EventArrays

H, W = 36, 64
FLOW = (3.0, -2.0)


def _segment(tmp_path, n_win=3):
    xs, ys, ts, ps = [], [], [], []
    wins = []
    for k in range(n_win):
        x, y, t, _ = moving_dots(height=H, width=W, flow_px=FLOW, seed=k, duration_us=33_333)
        xs.append(x.numpy())
        ys.append(y.numpy())
        ts.append(t.numpy().astype(np.int64) + 33_333 * k)
        ps.append(np.random.default_rng(k).integers(0, 2, x.numel()))
        wins.append(Window(k, 33_333 * k, 33_333 * (k + 1), 33_333 * (k - 1) if k else None))
    order = np.argsort(np.concatenate(ts), kind="stable")
    cat = [np.concatenate(a)[order] for a in (xs, ys, ts, ps)]
    arr = EventArrays(cat[0].astype(np.uint16), cat[1].astype(np.uint16), cat[2],
                      cat[3].astype(np.int8), source="fake")
    rng = np.random.default_rng(5)
    tf = np.zeros((n_win, 2, H, W), np.float32)
    tf[:, 0], tf[:, 1] = FLOW
    tf += rng.normal(0, 0.3, tf.shape).astype(np.float32)
    tv = rng.random((n_win, H, W)) > 0.25
    p = tmp_path / "teacher.h5"
    write(p, {"flow": tf, "valid": tv, "valid_literal": tv}, {"unit": "x", "frames_sha256": "s"})
    lab = np.full((H, W), 1, np.uint8)
    lab[:12] = 2
    lab[12:16, :20] = 0
    lab[-1, -1] = 255
    from evflow.io.lars import class_masks
    masks = [class_masks(lab)] * n_win
    seg = em.Segment("lars_eval_ct100/x", em.ArraySlicer(arr), wins, list(range(n_win)), p,
                     masks)
    return seg, tf, tv, masks[0], arr


class Const:
    def __init__(self, u, v):
        self.f = np.zeros((2, H, W))
        self.f[0], self.f[1] = u, v

    def __call__(self, win, ev):
        return self.f


def test_teacher_metrics_equal_reference_per_slice(tmp_path):
    seg, tf, tv, cm, arr = _segment(tmp_path)
    pred = Const(2.0, -1.0)
    pw = em.evaluate([seg], lambda s: pred, height=H, width=W)
    for i, win in enumerate(seg.windows):
        ev = arr.window(win.t_start_us, win.t_end_us)
        eiw = np.zeros((H, W), bool)
        eiw[ev["y"].astype(int), ev["x"].astype(int)] = True
        for s in em.SLICES:
            cls = np.ones((H, W), bool) if s == "all" else cm[s]
            ref = metrics_ref.flow_metrics(pred.f, tf[i].astype(np.float64), valid_gt=tv[i],
                                           events_in_window=eiw, class_mask=cls)
            if ref["n_px"] == 0:
                assert np.isnan(pw[f"teacher_epe__{s}"][i])
                continue
            assert pw[f"teacher_epe__{s}"][i] == pytest.approx(ref["epe"], rel=1e-9)
            assert pw[f"teacher_npe_3__{s}"][i] == pytest.approx(ref["npe_3"], rel=1e-9)
            assert pw[f"n_px__{s}"][i] == ref["n_px"]
            assert pw[f"teacher_mask_coverage__{s}"][i] == pytest.approx(
                (tv[i] & cls).sum() / cls.sum())
            assert pw[f"slice_px__{s}"][i] == cls.sum()
    assert pw["slice_px__all"][0] == H * W
    # partition: class slices + ignore cover the frame
    assert sum(pw[f"slice_px__{s}"][0] for s in ("water", "sky", "obstacle")) == H * W - 1


def test_fwl_rsat_zero_flow_is_one_and_true_flow_sharpens(tmp_path):
    seg, *_ = _segment(tmp_path)
    pz = em.evaluate([seg], lambda s: ZeroPredictor(H, W), height=H, width=W)
    for s in em.SLICES:
        ok = np.isfinite(pz[f"fwl__{s}"])
        assert ok.any()
        np.testing.assert_allclose(pz[f"fwl__{s}"][ok], 1.0)
        np.testing.assert_allclose(pz[f"rsat__{s}"][np.isfinite(pz[f"rsat__{s}"])], 1.0)
    pt = em.evaluate([seg], lambda s: Const(*FLOW), height=H, width=W)
    assert (pt["fwl__all"] > 1.0).all() and (pt["rsat__all"] < 1.0).all()
    assert (pt["teacher_epe__all"] < pz["teacher_epe__all"]).all()


def test_unsliced_windows_have_no_class_values(tmp_path):
    seg, *_ = _segment(tmp_path)
    seg.masks = [None] * len(seg.windows)
    pw = em.evaluate([seg], lambda s: Const(*FLOW), height=H, width=W)
    assert np.isfinite(pw["teacher_epe__all"]).all()
    assert np.isnan(pw["teacher_epe__water"]).all() and np.isnan(pw["slice_px__sky"]).all()


def test_cluster_ci_and_summary(tmp_path):
    v = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 5.0])
    g = np.array([0, 0, 1, 1, 1, 2])
    lo, hi = em.cluster_ci(v, g, resamples=2000, seed=0)
    assert lo <= v.mean() <= hi and lo >= 1.0 and hi <= 5.0
    # one cluster -> degenerate interval at the mean
    assert em.cluster_ci(np.array([2.0, 4.0]), np.array([7, 7])) == (3.0, 3.0)
    seg, *_ = _segment(tmp_path)
    pw = em.evaluate([seg], lambda s: Const(*FLOW), height=H, width=W)
    s = em.summarise_slice(pw, "water", em.T3A_KEYS, cluster=True, block_seconds=1.0,
                           resamples=200, ci=0.95, seed=0)
    assert s["n_windows"] == 3 and s["metrics"]["teacher_epe"] is not None
    assert set(s["metrics"]) == set(em.T3A_KEYS) and "epe" not in s["metrics"]
    s = em.summarise_slice(pw, "all", em.T3B_KEYS, cluster=False, block_seconds=1.0,
                           resamples=200, ci=0.95, seed=0)
    assert s["ci95"]["fwl"][0] <= s["metrics"]["fwl"] <= s["ci95"]["fwl"][1]


def test_unit_windows_rule():
    t = np.arange(6) * 1000
    wins, k = em.unit_windows(6, t, [2, 3, 4])
    assert [(w.t_start_us, w.t_end_us, w.prev_start_us) for w in wins] == [
        (2000, 3000, 1000), (3000, 4000, 2000), (4000, 5000, 3000)]
    with pytest.raises(ValueError):
        em.unit_windows(6, t, [5])


def test_rsat_guard_on_too_few_events():
    ev = {"x": np.array([1]), "y": np.array([1]), "t": np.array([5]), "p": np.array([1])}
    assert np.isnan(em.window_rsat(ev, np.zeros((2, H, W)), Window(0, 0, 10, None), H, W, None))
    assert torch.is_tensor(torch.zeros(1))
