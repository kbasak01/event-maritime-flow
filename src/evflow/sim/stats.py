"""Event statistics for simulator fidelity and v2e calibration (IMPLEMENTATION_PLAN §1.6).

Computed identically for real and synthetic events over the same windows and blocks (D-027):

| stat              | sample                                             | unit / scale        |
|-------------------|----------------------------------------------------|---------------------|
| `rate_per_window` | events/s in each window (APS dt=1)                 | log10 ev/s          |
| `on_fraction`     | ON / all in each non-empty window                  | fraction            |
| `pixel_rate`      | per active pixel per block: (count + 1) / block_s  | log10 ev/s          |
| `iei`             | per-pixel inter-event interval (any polarity)      | log10 µs (≥ 1 µs)   |
| `autocorr`        | count image, mean corr at lags 1–3 px, both axes   | correlation         |
| `hot_fraction`    | per block: pixels with rate > 10 × median active   | fraction of W·H     |

Every distribution is histogrammed on the fixed edges ``EDGES`` (samples outside are clipped into
the end bins), so each Wasserstein-1 distance is recomputable from saved histograms. The
calibration objective is the mean over stats of ``W1(real, syn) / std(real)`` — normalised
because the stats have different units (D-027).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import wasserstein_distance

STATS = ("rate_per_window", "on_fraction", "pixel_rate", "iei", "autocorr", "hot_fraction")
EDGES = {
    "rate_per_window": np.linspace(0.0, 9.0, 181),
    "on_fraction": np.linspace(0.0, 1.0, 101),
    "pixel_rate": np.linspace(0.0, 6.0, 121),
    "iei": np.linspace(0.0, 8.0, 161),
    "autocorr": np.linspace(-1.0, 1.0, 201),
    "hot_fraction": np.linspace(0.0, 0.05, 101),
}
LOG_FLOOR = 0.0            # log10 rate assigned to an empty window (1 ev/s, the lowest bin)
AUTOCORR_LAGS = (1, 2, 3)
HOT_FACTOR = 10.0


def _bounds(t: np.ndarray, wins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wins = np.asarray(wins, np.int64)
    return (np.searchsorted(t, wins[:, 0], side="left"),
            np.searchsorted(t, wins[:, 1], side="left"))


def count_image(e: dict, width: int, height: int, sl: slice = slice(None)) -> np.ndarray:
    idx = e["y"][sl].astype(np.int64) * width + e["x"][sl].astype(np.int64)
    return np.bincount(idx, minlength=width * height).reshape(height, width).astype(np.float64)


def autocorr(c: np.ndarray, lags=AUTOCORR_LAGS) -> float | None:
    """Mean Pearson correlation of the image with itself shifted by each lag, along x and y."""
    vals = []
    for lag in lags:
        for a, b in ((c[:, :-lag], c[:, lag:]), (c[:-lag, :], c[lag:, :])):
            a, b = a.ravel(), b.ravel()
            if a.std() == 0 or b.std() == 0:
                return None
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals))


def window_stats(e: dict, wins: np.ndarray, width: int, height: int) -> dict[str, np.ndarray]:
    lo, hi = _bounds(e["t"], wins)
    n = hi - lo
    dur_s = (np.asarray(wins)[:, 1] - np.asarray(wins)[:, 0]) / 1e6
    rate = np.where(n > 0, np.log10(np.maximum(n, 1) / dur_s), LOG_FLOOR)
    on = np.array([e["p"][a:b].sum() / (b - a) for a, b in zip(lo, hi, strict=True) if b > a],
                  np.float64)
    ac = [autocorr(count_image(e, width, height, slice(a, b)))
          for a, b in zip(lo, hi, strict=True) if b > a]
    return {"rate_per_window": rate.astype(np.float64), "on_fraction": on,
            "autocorr": np.array([v for v in ac if v is not None], np.float64)}


def block_stats(e: dict, blocks: np.ndarray, width: int, height: int,
                hot_factor: float = HOT_FACTOR) -> dict[str, np.ndarray]:
    lo, hi = _bounds(e["t"], blocks)
    rates, hot = [], []
    for (a, b), (t0, t1) in zip(zip(lo, hi, strict=True), np.asarray(blocks), strict=True):
        c = count_image(e, width, height, slice(a, b)).ravel()
        dur = (t1 - t0) / 1e6
        act = c[c > 0]
        if act.size == 0:
            hot.append(0.0)
            continue
        rates.append(np.log10((act + 1) / dur))
        hot.append(float(np.count_nonzero(act > hot_factor * np.median(act)) / c.size))
    return {"pixel_rate": np.concatenate(rates) if rates else np.zeros(0),
            "hot_fraction": np.array(hot, np.float64)}


def inter_event_intervals(e: dict, width: int) -> np.ndarray:
    """log10 of per-pixel inter-event intervals in µs (any polarity; 0 µs clipped to 1 µs)."""
    if len(e["t"]) < 2:
        return np.zeros(0)
    pix = e["y"].astype(np.int64) * width + e["x"].astype(np.int64)
    order = np.lexsort((e["t"], pix))
    p, t = pix[order], e["t"][order]
    same = p[1:] == p[:-1]
    d = (t[1:] - t[:-1])[same]
    return np.log10(np.maximum(d, 1).astype(np.float64))


def histogram(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float64), edges[0], edges[-1])
    h, _ = np.histogram(x, bins=edges)
    return h.astype(np.int64)


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[1:] + edges[:-1])


def w1_hist(ha: np.ndarray, hb: np.ndarray, edges: np.ndarray) -> float:
    if ha.sum() == 0 or hb.sum() == 0:
        return float("nan")
    c = _centers(edges)
    return float(wasserstein_distance(c, c, ha, hb))


def hist_std(h: np.ndarray, edges: np.ndarray) -> float:
    c = _centers(edges)
    w = h / h.sum()
    m = float((w * c).sum())
    return float(np.sqrt((w * (c - m) ** 2).sum()))


def objective(real: dict, syn: dict, edges: dict = EDGES) -> tuple[float, dict[str, float]]:
    """Mean over stats of W1(real, syn) / std(real) (std floored at one bin width)."""
    per = {}
    for k in real:
        width = float(edges[k][1] - edges[k][0])
        per[k] = w1_hist(real[k], syn[k], edges[k]) / max(hist_std(real[k], edges[k]), width)
    return float(np.mean(list(per.values()))), per


def slice_stats(e: dict, wins: np.ndarray, blocks: np.ndarray, width: int, height: int) -> dict:
    """Histograms of all §1.6 stats for one slice of events (absolute µs, sorted by t)."""
    samples = {**window_stats(e, wins, width, height), **block_stats(e, blocks, width, height),
               "iei": inter_event_intervals(e, width)}
    hist = {k: histogram(samples[k], EDGES[k]) for k in STATS}
    return {"hist": hist, "n_samples": {k: int(hist[k].sum()) for k in STATS},
            "mean": {k: (float(np.mean(samples[k])) if samples[k].size else None)
                     for k in STATS},
            "n_events": int(len(e["t"]))}


def count_ncc(real: dict, syn: dict, wins: np.ndarray, width: int, height: int,
              blur_sigma: float = 1.0) -> np.ndarray:
    """Per-window normalised cross-correlation of count images (both polarities), each blurred
    with a Gaussian of ``blur_sigma`` px; NaN where either image is constant."""
    lr, hr = _bounds(real["t"], wins)
    ls, hs = _bounds(syn["t"], wins)
    out = np.full(len(wins), np.nan)
    for i in range(len(wins)):
        a = count_image(real, width, height, slice(lr[i], hr[i]))
        b = count_image(syn, width, height, slice(ls[i], hs[i]))
        if blur_sigma > 0:
            a = cv2.GaussianBlur(a, (0, 0), blur_sigma)
            b = cv2.GaussianBlur(b, (0, 0), blur_sigma)
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            out[i] = float((a * b).sum() / den)
    return out
