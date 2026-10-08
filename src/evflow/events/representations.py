"""Torch event representations (GPU-capable, ``index_put_`` with accumulate).

Same definitions as ``representations_ref.py`` (the numpy reference they are tested against):
integer x/y land on one pixel, float x/y are splatted bilinearly; channel index = polarity
(0 OFF, 1 ON); ``t_range`` is explicit.
"""

from __future__ import annotations

import torch


def _spatial_taps(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype):
    if not torch.is_floating_point(x) and not torch.is_floating_point(y):
        yield x.long(), y.long(), torch.ones(x.shape, dtype=dtype, device=x.device)
        return
    x = x.to(dtype)
    y = y.to(dtype)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    for dx in (0, 1):
        for dy in (0, 1):
            xi = x0 + dx
            yi = y0 + dy
            w = (1.0 - (xi - x).abs()) * (1.0 - (yi - y).abs())
            yield xi.long(), yi.long(), w


def _splat(img: torch.Tensor, c, xi, yi, w) -> None:
    n_c, h, wd = img.shape
    ok = (c >= 0) & (c < n_c) & (xi >= 0) & (xi < wd) & (yi >= 0) & (yi < h)
    img.index_put_((c[ok], yi[ok], xi[ok]), w[ok], accumulate=True)


def count_image(x, y, p, *, height: int, width: int, dtype=torch.float32) -> torch.Tensor:
    """Per-polarity event counts, shape (2, H, W): channel 0 = OFF, 1 = ON."""
    img = torch.zeros((2, height, width), dtype=dtype, device=x.device)
    p = p.long()
    for xi, yi, w in _spatial_taps(x, y, dtype):
        _splat(img, p, xi, yi, w)
    return img


def _normalize_time(t, t_range, scale, dtype):
    t0, t1 = t_range
    if not t1 > t0:
        raise ValueError(f"t_range must satisfy t_end > t_start, got {t_range}")
    # subtract in int64 first so float32 keeps µs resolution on large absolute times
    return scale * (t - int(t0)).to(dtype) / float(t1 - t0)


def voxel_grid(
    x, y, t, p, *, bins: int, height: int, width: int, t_range: tuple[int, int], normalize: bool,
    dtype=torch.float32,
) -> torch.Tensor:
    """Voxel grid (B, H, W), see ``representations_ref.voxel_grid``."""
    vox = torch.zeros((bins, height, width), dtype=dtype, device=x.device)
    if t.numel() == 0:
        _normalize_time(t, t_range, 1.0, dtype)
        return vox
    tn = _normalize_time(t, t_range, bins - 1, dtype)
    t0 = torch.floor(tn)
    pol = 2.0 * p.to(dtype) - 1.0
    for xi, yi, ws in _spatial_taps(x, y, dtype):
        for dt in (0, 1):
            ti = t0 + dt
            wt = torch.clamp(1.0 - (ti - tn).abs(), min=0.0)
            _splat(vox, ti.long(), xi, yi, pol * ws * wt)
    if normalize:
        nz = vox != 0
        n = int(nz.sum())
        if n > 0:
            vals = vox[nz]
            mean = vals.mean()
            std = vals.std() if n > 1 else torch.zeros((), dtype=dtype, device=vox.device)
            vox[nz] = (vals - mean) / std if std > 0 else vals - mean
    return vox


def time_surface(
    x, y, t, p, *, height: int, width: int, t_range: tuple[int, int], tau_us: float,
    dtype=torch.float32,
) -> torch.Tensor:
    """Per-polarity exponential time surface (2, H, W), see ``representations_ref.time_surface``."""
    t0, t1 = t_range
    if not t1 > t0:
        raise ValueError(f"t_range must satisfy t_end > t_start, got {t_range}")
    if torch.is_floating_point(x) or torch.is_floating_point(y):
        xi = torch.round(x.double()).long()
        yi = torch.round(y.double()).long()
    else:
        xi, yi = x.long(), y.long()
    p = p.long()
    ok = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    flat = (p[ok] * height + yi[ok]) * width + xi[ok]
    # age relative to t_end in int64, then max over events = min age
    neg_age = -(int(t1) - t[ok]).to(torch.float64)
    last = torch.full((2 * height * width,), -float("inf"), dtype=torch.float64, device=x.device)
    last.scatter_reduce_(0, flat, neg_age, reduce="amax")
    out = torch.zeros_like(last)
    has = torch.isfinite(last)
    out[has] = torch.exp(last[has] / float(tau_us))
    return out.view(2, height, width).to(dtype)
