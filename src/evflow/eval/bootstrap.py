"""Moving-block bootstrap CIs over evaluation windows (plan §1.6, configs/evflow.yaml
``metrics.bootstrap``: block = 1 s of windows, 1,000 resamples, 95 %).

Künsch's moving-block bootstrap: a resample concatenates ``ceil(n / L)`` blocks of ``L``
consecutive windows whose starts are drawn uniformly from ``0 .. n - L``, truncated to ``n``
values; the CI is the percentile interval of the resampled statistic. Consecutive windows overlap
or share scene content, so i.i.d. resampling would understate the uncertainty.

Paired (Phase 6): arm-vs-arm statistics (gap, closure) apply the *same* resampled window indices
to every series, so content shared by the arms cancels. Stratified: pooled rows over several
sequences resample blocks independently within each sequence and average per-sequence means.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

Stat = Callable[[Mapping[str, np.ndarray]], np.ndarray | float]


def block_length(window_starts_us: np.ndarray, block_seconds: float) -> int:
    """Windows per block: ``block_seconds`` divided by the median start-to-start stride, >= 1."""
    t = np.asarray(window_starts_us, np.int64)
    if t.size < 2:
        return 1
    stride_us = float(np.median(np.diff(t)))
    if stride_us <= 0:
        raise ValueError("window starts must be increasing")
    return max(1, int(round(block_seconds * 1e6 / stride_us)))


def _draw(rng: np.random.Generator, n: int, block: int, resamples: int) -> np.ndarray:
    L = min(max(1, int(block)), n)
    k = math.ceil(n / L)
    starts = rng.integers(0, n - L + 1, size=(resamples, k))
    return (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(resamples, -1)[:, :n]


def block_indices(n: int, *, block: int, resamples: int = 1000, seed: int = 0) -> np.ndarray:
    """(resamples, n) window indices of the moving-block bootstrap."""
    if n <= 0:
        raise ValueError("n must be positive")
    return _draw(np.random.default_rng(seed), n, block, resamples)


def _interval(samples: np.ndarray, ci: float) -> tuple[float, float]:
    a = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [a, 1.0 - a])
    return float(lo), float(hi)


def moving_block_ci(values: np.ndarray, *, block: int, resamples: int = 1000, ci: float = 0.95,
                    seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the mean of ``values`` (NaN-free 1-D) under the moving-block bootstrap."""
    v = np.asarray(values, np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite (drop skipped windows first)")
    idx = block_indices(v.size, block=block, resamples=resamples, seed=seed)
    return _interval(v[idx].mean(axis=1), ci)


def _check_series(series: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    s = {k: np.asarray(v, np.float64) for k, v in series.items()}
    sizes = {v.shape for v in s.values()}
    if not s or len(sizes) != 1 or next(iter(sizes))[0] == 0 or len(next(iter(sizes))) != 1:
        raise ValueError("series must be non-empty 1-D arrays of one length (paired windows)")
    if not all(np.all(np.isfinite(v)) for v in s.values()):
        raise ValueError("series must be finite (keep windows scored in every series)")
    return s


def paired_block_ci(series: Mapping[str, np.ndarray], stat: Stat, *, block: int,
                    resamples: int = 1000, ci: float = 0.95,
                    seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) of ``stat`` under a paired moving-block bootstrap.

    ``series`` are per-window values on the same windows; ``stat`` maps a dict of arrays whose
    last axis is the window axis to the statistic along that axis (it is called once on the 1-D
    series for the point estimate and once on (resamples, n) arrays)."""
    s = _check_series(series)
    n = next(iter(s.values())).size
    idx = block_indices(n, block=block, resamples=resamples, seed=seed)
    point = np.asarray(stat(s), np.float64).item()
    lo, hi = _interval(np.asarray(stat({k: v[idx] for k, v in s.items()}), np.float64), ci)
    return point, lo, hi


def stratified_paired_ci(strata: Sequence[Mapping[str, np.ndarray]],
                         stat: Callable[[list[Mapping[str, np.ndarray]]], np.ndarray | float], *,
                         blocks: Sequence[int], resamples: int = 1000, ci: float = 0.95,
                         seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap over several strata (sequences), each resampled independently with its
    own block length from one generator; ``stat`` receives the list of per-stratum dicts."""
    if len(strata) != len(blocks) or not strata:
        raise ValueError("one block length per stratum, at least one stratum")
    checked = [_check_series(s) for s in strata]
    rng = np.random.default_rng(seed)
    resampled = []
    for s, b in zip(checked, blocks, strict=True):
        idx = _draw(rng, next(iter(s.values())).size, b, resamples)
        resampled.append({k: v[idx] for k, v in s.items()})
    point = np.asarray(stat(checked), np.float64).item()
    lo, hi = _interval(np.asarray(stat(resampled), np.float64), ci)
    return point, lo, hi
