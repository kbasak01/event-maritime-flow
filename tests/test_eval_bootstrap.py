"""Paired and stratified moving-block bootstrap (Phase 6: gap, closure, arm-vs-arm CIs)."""

import numpy as np
import pytest

from evflow.eval.bootstrap import (
    block_indices,
    moving_block_ci,
    paired_block_ci,
    stratified_paired_ci,
)


def _gap(s):
    return s["a"].mean(-1) - s["b"].mean(-1)


def test_moving_block_ci_unchanged_by_refactor():
    # value captured from the Phase 4/5 implementation before block_indices was factored out
    v = np.sin(np.arange(257) * 0.37) + np.arange(257) * 0.01
    assert moving_block_ci(v, block=7, resamples=200, seed=3) == (0.9856060860508143,
                                                                   1.5265990219632264)


def test_block_indices_shape_and_contiguity():
    idx = block_indices(23, block=5, resamples=50, seed=0)
    assert idx.shape == (50, 23)
    assert idx.min() >= 0 and idx.max() < 23
    d = np.diff(idx[:, :5], axis=1)       # the first block is 5 consecutive windows
    assert np.all(d == 1)


def test_paired_identical_series_give_zero_gap():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    pt, lo, hi = paired_block_ci({"a": a, "b": a.copy()}, _gap, block=10)
    assert pt == lo == hi == 0.0


def test_paired_constant_offset_is_exact():
    rng = np.random.default_rng(1)
    b = rng.normal(size=400)
    pt, lo, hi = paired_block_ci({"a": b + 0.25, "b": b}, _gap, block=12)
    assert pt == pytest.approx(0.25) and lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_paired_ci_narrower_than_unpaired_for_shared_noise():
    rng = np.random.default_rng(2)
    shared = np.repeat(rng.normal(size=40), 10)
    a = shared + 0.1 + 0.01 * rng.normal(size=400)
    b = shared + 0.01 * rng.normal(size=400)
    _, lo, hi = paired_block_ci({"a": a, "b": b}, _gap, block=10)
    lo_a, hi_a = moving_block_ci(a, block=10)
    assert lo < 0.1 < hi
    assert (hi - lo) < 0.2 * (hi_a - lo_a)


def test_known_closure():
    rng = np.random.default_rng(3)
    real = 1.0 + 0.05 * rng.normal(size=500)
    fixed = real + 0.4
    rand = real + 0.1            # closes 75 % of the fixed gap

    def closure(s):
        return 1.0 - (s["rand"].mean(-1) - s["real"].mean(-1)) / (
            s["fixed"].mean(-1) - s["real"].mean(-1))

    pt, lo, hi = paired_block_ci({"real": real, "fixed": fixed, "rand": rand}, closure, block=20)
    assert pt == pytest.approx(0.75) and lo == pytest.approx(0.75) and hi == pytest.approx(0.75)


def test_stratified_one_stratum_equals_unstratified():
    rng = np.random.default_rng(4)
    s = {"a": rng.normal(size=200) + 0.3, "b": rng.normal(size=200)}
    flat = paired_block_ci(s, _gap, block=8, resamples=300, seed=5)
    strat = stratified_paired_ci([s], lambda ss: _gap(ss[0]), blocks=[8], resamples=300, seed=5)
    assert strat == flat


def test_stratified_mean_of_means():
    a1, b1 = np.full(100, 2.0), np.full(100, 1.0)      # gap 1
    a2, b2 = np.full(30, 0.5), np.full(30, 0.5)        # gap 0

    def pooled(ss):
        return np.mean([_gap(s) for s in ss], axis=0)

    pt, lo, hi = stratified_paired_ci([{"a": a1, "b": b1}, {"a": a2, "b": b2}], pooled,
                                      blocks=[10, 3])
    assert pt == lo == hi == pytest.approx(0.5)        # mean over sequences, not over windows


def test_paired_errors():
    with pytest.raises(ValueError):
        paired_block_ci({"a": np.ones(5), "b": np.ones(6)}, _gap, block=2)
    with pytest.raises(ValueError):
        paired_block_ci({"a": np.array([1.0, np.nan]), "b": np.ones(2)}, _gap, block=1)
    with pytest.raises(ValueError):
        stratified_paired_ci([{"a": np.ones(3), "b": np.ones(3)}], lambda ss: _gap(ss[0]),
                             blocks=[1, 2])
