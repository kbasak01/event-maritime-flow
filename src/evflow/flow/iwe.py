"""Torch image of warped events; same definitions as ``iwe_ref.py`` (tested equal to it)."""

from __future__ import annotations

import torch

from evflow.flow.iwe_ref import INTERPOLATIONS


def normalize_time(t: torch.Tensor, t_range, dtype=torch.float64) -> torch.Tensor:
    t0, t1 = t_range
    if not t1 > t0:
        raise ValueError(f"t_range must satisfy t_end > t_start, got {t_range}")
    return (t.long() - int(t0)).to(dtype) / float(t1 - t0)


def sample_flow(flow: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-event flow (N, 2) = flow[:, y, x] at each event's own pixel."""
    _, h, w = flow.shape
    if torch.is_floating_point(x) or torch.is_floating_point(y):
        x = torch.round(x.double())
        y = torch.round(y.double())
    xi = x.long().clamp(0, w - 1)
    yi = y.long().clamp(0, h - 1)
    return flow[:, yi, xi].T


def warp(x, y, tau, flow_ev, *, t_ref: float):
    dt = float(t_ref) - tau
    return x.to(tau.dtype) + dt * flow_ev[:, 0], y.to(tau.dtype) + dt * flow_ev[:, 1]


def splat(xw, yw, *, height: int, width: int, interpolation: str, weights=None) -> torch.Tensor:
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
    wts = torch.ones_like(xw) if weights is None else weights.to(xw.dtype)
    img = torch.zeros(height * width, dtype=xw.dtype, device=xw.device)
    if interpolation == "round":
        taps = [(torch.round(xw), torch.round(yw), wts)]
    else:
        x0, y0 = torch.floor(xw), torch.floor(yw)
        taps = []
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                w = (torch.clamp(1.0 - (xw - xi).abs(), min=0.0)
                     * torch.clamp(1.0 - (yw - yi).abs(), min=0.0))
                taps.append((xi, yi, wts * w))
    for xi, yi, w in taps:
        ok = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        idx = yi[ok].long() * width + xi[ok].long()
        img.index_put_((idx,), w[ok], accumulate=True)
    return img.view(height, width)


def iwe(x, y, t, flow, *, t_range, t_ref: float, height: int, width: int, interpolation: str,
        weights=None) -> torch.Tensor:
    """Image of events warped to ``t_ref`` by ``flow`` (H, W). Differentiable w.r.t. ``flow``
    for ``interpolation='bilinear'``."""
    tau = normalize_time(t, t_range, dtype=flow.dtype)
    xw, yw = warp(x, y, tau, sample_flow(flow, x, y), t_ref=t_ref)
    return splat(xw, yw, height=height, width=width, interpolation=interpolation, weights=weights)
