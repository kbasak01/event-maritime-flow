"""sim/fidelity on synthetic event streams (no data): window bookkeeping, identical streams,
shift and mirror controls, pooling."""

import numpy as np
import pytest

from evflow.sim import fidelity as F
from evflow.sim import stats as S

W, H = 64, 48
APS = np.arange(0, 2_000_001, 25_000, dtype=np.int64) + 10_000_000   # 81 frames, 40 fps
UNIT = {"aps": [0, 80], "owned_window_starts": [0, 79], "stats_from_window": 10}


def _moving_edges(seed=0, n_per_win=400):
    """A textured stripe pattern that moves 1 px per frame (edges co-locate per window)."""
    rng = np.random.default_rng(seed)
    t, x, y, p = [], [], [], []
    for k in range(len(APS) - 1):
        tt = np.sort(rng.integers(APS[k], APS[k + 1], n_per_win))
        xs = (rng.integers(0, 6, n_per_win) * 10 + k) % W
        t.append(tt)
        x.append(xs)
        y.append(rng.integers(0, H, n_per_win))
        p.append(rng.integers(0, 2, n_per_win))
    return {"t": np.concatenate(t), "x": np.concatenate(x).astype(np.uint16),
            "y": np.concatenate(y).astype(np.uint16), "p": np.concatenate(p).astype(np.int8)}


def test_owned_windows_skip_warmup_and_fit_unit():
    k = F.owned_window_starts(UNIT)
    assert k[0] == 10 and k[-1] == 79
    k4 = F.owned_window_starts(UNIT, dt=4)
    assert k4[-1] + 4 <= UNIT["aps"][1]
    chunk = {"aps": [90, 200], "owned_window_starts": [100, 196], "stats_from_window": 100}
    assert F.owned_window_starts(chunk)[0] == 100


def test_identical_streams_zero_distance_and_unit_ncc():
    e = _moving_edges()
    st = F.unit_stats(e, e, APS, UNIT, 500_000, W, H)
    d = F.distances(F.pool([st]))
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in d["w1"].values())
    n = F.unit_ncc(e, e, APS, UNIT, W, H, blur_sigma=1.0)
    assert np.nanmin(n["aligned"]) == pytest.approx(1.0)
    assert np.nanmedian(n["shift5"]) < 0.9 and np.nanmedian(n["mirror"]) < 0.9


def test_shift_control_pairs_exact_frames():
    e = _moving_edges()
    k = F.owned_window_starts(UNIT)
    wins = F.windows(APS, k)
    # syn = real delayed by exactly 1 frame: the 1-frame control must align perfectly
    delayed = {**e, "t": e["t"] + 25_000}
    n = S.count_ncc(e, delayed, wins[:-1], W, H, 1.0, syn_wins=F.windows(APS, k[:-1] + 1))
    assert np.nanmin(n) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        S.count_ncc(e, e, wins, W, H, 1.0, syn_wins=wins[:-1])


def test_restrict_and_span_identical_for_both():
    e = _moving_edges()
    syn = _moving_edges(seed=1)
    st = F.unit_stats(e, syn, APS, UNIT, 500_000, W, H)
    span = st["span_us"]
    assert span == [int(APS[10]), int(APS[80])]
    assert st["n_blocks"] == 3 and st["n_windows"] == 70
    r = F.restrict(e, *span)
    assert st["real"]["n_events"] == len(r["t"]) and r["t"].min() >= span[0]


def test_pool_adds_histograms_and_weights_means():
    e = _moving_edges()
    a = F.unit_stats(e, e, APS, UNIT, 500_000, W, H)
    p = F.pool([a, a])
    for k in S.STATS:
        assert np.array_equal(p["real"]["hist"][k], 2 * np.asarray(a["real"]["hist"][k]))
        if a["real"]["mean"][k] is not None:
            assert p["real"]["mean"][k] == pytest.approx(a["real"]["mean"][k])
    assert p["n_windows"] == 2 * a["n_windows"]


def test_ncc_summary_threshold_and_controls():
    v = {"aligned": [np.array([0.6, 0.7, np.nan])], "shift1": [np.array([0.5])],
         "shift5": [np.array([0.2])], "mirror": [np.array([0.05])]}
    s = F.ncc_summary(v, 0.44)
    assert s["pass"] and s["n_nan"] == 1 and s["median"] == pytest.approx(0.65)
    assert not s["controls"]["shift1"]["below_threshold"]
    assert s["controls"]["mirror"]["below_threshold"]
    assert s["controls"]["shift5"]["aligned_minus_control"] == pytest.approx(0.45)
