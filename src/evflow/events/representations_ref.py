"""Pure-numpy reference event representations (IMPLEMENTATION_PLAN §1.5).

The torch versions in ``representations.py`` are tested equal to these (float64, 1e-6).

Conventions shared by every function here:
- ``x``/``y`` integer pixels, or float sub-pixel coordinates (e.g. DSEC rectified). Float
  coordinates are splatted bilinearly onto the 4 neighbouring pixels (``floor`` and ``floor + 1``);
  integer coordinates land on exactly one pixel. Weight falling outside the image is dropped.
- ``p`` in {0 = OFF, 1 = ON}. Per-polarity outputs use channel index = polarity value.
- ``t`` int64 µs. ``t_range = (t_start, t_end)`` is passed explicitly: no hidden window.
"""

from __future__ import annotations

import numpy as np


def _is_int(a: np.ndarray) -> bool:
    return np.issubdtype(np.asarray(a).dtype, np.integer)


def _spatial_taps(x, y):
    """Yield (xi, yi, w) taps: one tap for integer coords, four bilinear taps for float coords."""
    if _is_int(x) and _is_int(y):
        yield np.asarray(x, np.int64), np.asarray(y, np.int64), np.ones(len(x))
        return
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x0 = np.floor(x)
    y0 = np.floor(y)
    for dx in (0, 1):
        for dy in (0, 1):
            xi = x0 + dx
            yi = y0 + dy
            w = (1.0 - np.abs(xi - x)) * (1.0 - np.abs(yi - y))
            yield xi.astype(np.int64), yi.astype(np.int64), w


def _splat(img: np.ndarray, c: np.ndarray, xi, yi, w) -> None:
    """Accumulate weights ``w`` at (channel c, yi, xi); taps outside the volume are dropped."""
    n_c, h, wd = img.shape
    ok = (c >= 0) & (c < n_c) & (xi >= 0) & (xi < wd) & (yi >= 0) & (yi < h)
    np.add.at(img, (c[ok], yi[ok], xi[ok]), w[ok])


def count_image(x, y, p, *, height: int, width: int) -> np.ndarray:
    """Per-polarity event counts, shape (2, H, W): channel 0 = OFF, 1 = ON."""
    img = np.zeros((2, height, width), dtype=np.float64)
    p = np.asarray(p, np.int64)
    for xi, yi, w in _spatial_taps(x, y):
        _splat(img, p, xi, yi, w)
    return img


def normalize_time(t, t_range: tuple[int, int], scale: float) -> np.ndarray:
    t0, t1 = t_range
    if not t1 > t0:
        raise ValueError(f"t_range must satisfy t_end > t_start, got {t_range}")
    return scale * (np.asarray(t, np.float64) - t0) / float(t1 - t0)


def voxel_grid(
    x, y, t, p, *, bins: int, height: int, width: int, t_range: tuple[int, int], normalize: bool
) -> np.ndarray:
    """Voxel grid (B, H, W): signed polarity (2p - 1), bilinear in time over ``bins`` bins whose
    centres are at ``t_start`` ... ``t_end`` (E-RAFT / DSEC ``VoxelGrid``:
    ``t_norm = (B - 1) (t - t_start) / (t_end - t_start)``). With ``normalize``, non-zero cells are
    shifted/scaled to mean 0, std 1 (unbiased std, as torch.std), as in E-RAFT."""
    vox = np.zeros((bins, height, width), dtype=np.float64)
    if len(np.asarray(t)) == 0:
        normalize_time(np.zeros(0), t_range, 1.0)  # still validate the range
        return vox
    tn = normalize_time(t, t_range, bins - 1)
    t0 = np.floor(tn)
    pol = 2.0 * np.asarray(p, np.float64) - 1.0
    for xi, yi, ws in _spatial_taps(x, y):
        for dt in (0, 1):
            ti = t0 + dt
            wt = np.maximum(0.0, 1.0 - np.abs(ti - tn))
            _splat(vox, ti.astype(np.int64), xi, yi, pol * ws * wt)
    if normalize:
        nz = vox != 0
        if nz.any():
            mean = vox[nz].mean()
            std = vox[nz].std(ddof=1) if nz.sum() > 1 else 0.0
            vox[nz] = (vox[nz] - mean) / std if std > 0 else vox[nz] - mean
    return vox


def time_surface(
    x, y, t, p, *, height: int, width: int, t_range: tuple[int, int], tau_us: float
) -> np.ndarray:
    """Per-polarity exponential time surface (2, H, W): ``exp(-(t_end - t_last) / tau)`` where
    ``t_last`` is the latest event of that polarity at the pixel in the window; 0 where none.
    Float coordinates are rounded to the nearest pixel (half to even, as torch.round)."""
    t0, t1 = t_range
    if not t1 > t0:
        raise ValueError(f"t_range must satisfy t_end > t_start, got {t_range}")
    if _is_int(x) and _is_int(y):
        xi = np.asarray(x, np.int64)
        yi = np.asarray(y, np.int64)
    else:
        xi = np.round(np.asarray(x, np.float64)).astype(np.int64)
        yi = np.round(np.asarray(y, np.float64)).astype(np.int64)
    p = np.asarray(p, np.int64)
    ok = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    last = np.full((2, height, width), -np.inf)
    np.maximum.at(last, (p[ok], yi[ok], xi[ok]), np.asarray(t, np.float64)[ok])
    out = np.zeros((2, height, width), dtype=np.float64)
    has = np.isfinite(last)
    out[has] = np.exp(-(float(t1) - last[has]) / float(tau_us))
    return out
