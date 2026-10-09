"""sim/stats: §1.6 event statistics, histogram W1, objective, count NCC on hand-built events."""

import numpy as np
import pytest

from evflow.sim import stats as S


def ev(t, x, y, p):
    return {"t": np.asarray(t, np.int64), "x": np.asarray(x, np.uint16),
            "y": np.asarray(y, np.uint16), "p": np.asarray(p, np.int8)}


W, H = 8, 6


def test_window_rate_and_on_fraction():
    # window 0: 4 events (3 ON) in 10 ms; window 1: 2 events (0 ON)
    e = ev([0, 1, 2, 3, 10_000, 15_000], [0] * 6, [0] * 6, [1, 1, 1, 0, 0, 0])
    wins = np.array([[0, 10_000], [10_000, 20_000]])
    r = S.window_stats(e, wins, W, H)
    np.testing.assert_allclose(r["rate_per_window"], np.log10([400.0, 200.0]))
    np.testing.assert_allclose(r["on_fraction"], [0.75, 0.0])


def test_empty_window_skipped_for_on_fraction_and_rate_floor():
    e = ev([0, 1], [0, 1], [0, 0], [1, 0])
    wins = np.array([[0, 10_000], [10_000, 20_000]])
    r = S.window_stats(e, wins, W, H)
    assert r["on_fraction"].tolist() == [0.5]           # empty window has no ON fraction
    assert r["rate_per_window"][1] == S.LOG_FLOOR       # empty window -> floor, not -inf


def test_autocorr_checkerboard_and_smooth():
    # vertical stripes of width 1 in count image: lag-1 horizontal corr = -1, vertical = +1
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    m = (xs % 2 == 0)
    e = ev(np.zeros(m.sum()), xs[m], ys[m], np.ones(m.sum()))
    c = S.count_image(e, W, H)
    ac = S.autocorr(c, lags=(1,))
    assert ac == pytest.approx(0.0)                      # mean of (-1 horizontal, +1 vertical)
    assert S.autocorr(np.ones((H, W)), lags=(1,)) is None   # constant image: undefined


def test_iei_per_pixel():
    # pixel (0,0): t 0, 10, 110 -> IEI 10, 100; pixel (1,0): t 5, 1005 -> 1000
    e = ev([0, 5, 10, 110, 1005], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1] * 5)
    iei = S.inter_event_intervals(e, W)
    np.testing.assert_allclose(np.sort(iei), np.log10([10, 100, 1000]))


def test_iei_zero_interval_clipped_to_1us():
    e = ev([7, 7], [0, 0], [0, 0], [1, 0])
    np.testing.assert_allclose(S.inter_event_intervals(e, W), [0.0])


def test_block_pixel_rates_and_hot_fraction():
    # 1 s block: pixel (0,0) 1000 events, pixels (1..4, 0) 10 events each; others none
    t = np.concatenate([np.arange(1000) * 900, np.repeat(np.arange(10) * 90_000, 4)])
    x = np.concatenate([np.zeros(1000), np.tile([1, 2, 3, 4], 10)])
    order = np.argsort(t, kind="stable")
    e = ev(t[order], x[order], np.zeros(len(t))[order], np.ones(len(t))[order])
    r = S.block_stats(e, np.array([[0, 1_000_000]]), W, H, hot_factor=10.0)
    rates = np.sort(r["pixel_rate"])
    np.testing.assert_allclose(rates, np.log10([11, 11, 11, 11, 1001]))  # log10(count+1), active
    assert r["hot_fraction"].tolist() == [pytest.approx(1 / (W * H))]    # 1000 > 10 * median(10)


def test_histogram_w1_identity_shift_and_normalisation():
    edges = np.linspace(0, 10, 101)
    a = np.random.default_rng(0).normal(5, 1, 20000)
    ha, hb = S.histogram(a, edges), S.histogram(a + 1.0, edges)
    assert S.w1_hist(ha, ha, edges) == 0.0
    assert S.w1_hist(ha, hb, edges) == pytest.approx(1.0, abs=0.02)
    assert S.hist_std(ha, edges) == pytest.approx(1.0, abs=0.02)
    # out-of-range samples are clipped into the end bins and counted
    h = S.histogram(np.array([-5.0, 50.0, 5.0]), edges)
    assert h.sum() == 3 and h[0] == 1 and h[-1] == 1


def test_objective_mean_of_normalised_w1():
    edges = {"a": np.linspace(0, 10, 101), "b": np.linspace(0, 100, 101)}
    rng = np.random.default_rng(1)
    real = {"a": S.histogram(rng.normal(5, 1, 50000), edges["a"]),
            "b": S.histogram(rng.normal(50, 10, 50000), edges["b"])}
    syn = {"a": S.histogram(rng.normal(6, 1, 50000), edges["a"]),     # shift 1 std
           "b": S.histogram(rng.normal(50, 10, 50000), edges["b"])}    # identical
    obj, per = S.objective(real, syn, edges)
    assert per["a"] == pytest.approx(1.0, abs=0.05) and per["b"] < 0.05
    assert obj == pytest.approx((per["a"] + per["b"]) / 2)


def test_ncc_identical_shifted_and_empty():
    rng = np.random.default_rng(2)
    n = 3000
    e = ev(np.sort(rng.integers(0, 20_000, n)), rng.integers(0, W, n), rng.integers(0, H, n),
           rng.integers(0, 2, n))
    wins = np.array([[0, 10_000], [10_000, 20_000]])
    nc = S.count_ncc(e, e, wins, W, H, blur_sigma=0.0)
    np.testing.assert_allclose(nc, [1.0, 1.0])
    empty = ev([], [], [], [])
    assert np.isnan(S.count_ncc(e, empty, wins, W, H, blur_sigma=0.0)).all()


def test_slice_stats_end_to_end_keys():
    rng = np.random.default_rng(3)
    n = 20000
    e = ev(np.sort(rng.integers(0, 2_000_000, n)), rng.integers(0, W, n), rng.integers(0, H, n),
           rng.integers(0, 2, n))
    wins = np.stack([np.arange(0, 2_000_000, 20_000), np.arange(20_000, 2_000_001, 20_000)], 1)
    blocks = np.array([[0, 1_000_000], [1_000_000, 2_000_000]])
    st = S.slice_stats(e, wins, blocks, W, H)
    assert set(st["hist"]) == set(S.STATS)
    for k in S.STATS:
        assert st["hist"][k].shape == (len(S.EDGES[k]) - 1,)
        assert st["n_samples"][k] == st["hist"][k].sum()
