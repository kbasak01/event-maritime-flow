"""Moving-block bootstrap CI of a mean over evaluation windows (plan §1.6, configs/evflow.yaml
``metrics.bootstrap``: block = 1 s of windows, 1,000 resamples, 95 %).

Künsch's moving-block bootstrap: a resample concatenates ``ceil(n / L)`` blocks of ``L``
consecutive windows whose starts are drawn uniformly from ``0 .. n - L``, truncated to ``n``
values; the CI is the percentile interval of the resampled means. Consecutive windows overlap or
share scene content, so i.i.d. resampling would understate the uncertainty.
"""

from __future__ import annotations

import math

import numpy as np


def block_length(window_starts_us: np.ndarray, block_seconds: float) -> int:
    """Windows per block: ``block_seconds`` divided by the median start-to-start stride, >= 1."""
    t = np.asarray(window_starts_us, np.int64)
    if t.size < 2:
        return 1
    stride_us = float(np.median(np.diff(t)))
    if stride_us <= 0:
        raise ValueError("window starts must be increasing")
    return max(1, int(round(block_seconds * 1e6 / stride_us)))


def moving_block_ci(values: np.ndarray, *, block: int, resamples: int = 1000, ci: float = 0.95,
                    seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the mean of ``values`` (NaN-free 1-D) under the moving-block bootstrap."""
    v = np.asarray(values, np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite (drop skipped windows first)")
    n = v.size
    L = min(max(1, int(block)), n)
    k = math.ceil(n / L)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - L + 1, size=(resamples, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(resamples, -1)[:, :n]
    means = v[idx].mean(axis=1)
    a = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [a, 1.0 - a])
    return float(lo), float(hi)
