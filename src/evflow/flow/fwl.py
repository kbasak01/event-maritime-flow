"""Torch FWL and RSAT (self-supervised flow metrics; T3b uses only these).

Definitions in ``iwe_ref.py`` (numpy reference, tested equal). FWL > 1 is better and zero flow
gives exactly 1; RSAT < 1 is better and zero flow gives exactly 1. Neither is an accuracy metric.
"""

from __future__ import annotations

import torch

from evflow.flow.iwe import INTERPOLATIONS, iwe, normalize_time, sample_flow, splat, warp


@torch.no_grad()
def fwl(x, y, t, flow, *, t_range, t_ref: float, height: int, width: int,
        interpolation: str) -> float:
    a = dict(t_range=t_range, t_ref=t_ref, height=height, width=width,
             interpolation=interpolation)
    img = iwe(x, y, t, flow, **a)
    img0 = iwe(x, y, t, torch.zeros_like(flow), **a)
    return float(img.var(unbiased=False) / img0.var(unbiased=False))


def _rsat_term(x, y, tau, p, flow, a) -> float:
    xw, yw = warp(x, y, tau, sample_flow(flow, x, y), t_ref=a["t_ref"])
    s = dict(height=a["height"], width=a["width"], interpolation=a["interpolation"])
    total = 0.0
    hit = torch.zeros((a["height"], a["width"]), dtype=torch.bool, device=flow.device)
    for pol in (1, 0):
        m = p == pol
        cnt = splat(xw[m], yw[m], **s)
        ts = splat(xw[m], yw[m], weights=tau[m], **s) / (cnt + 1e-9)
        total += float((ts**2).sum())
        hit |= cnt > 0
    return total / float(hit.sum())


@torch.no_grad()
def rsat(x, y, t, p, flow, *, t_range, t_ref: float, height: int, width: int,
         interpolation: str) -> float:
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
    a = dict(t_ref=t_ref, height=height, width=width, interpolation=interpolation)
    tau = normalize_time(t, t_range, dtype=flow.dtype)
    return _rsat_term(x, y, tau, p, flow, a) / _rsat_term(x, y, tau, p, torch.zeros_like(flow), a)
