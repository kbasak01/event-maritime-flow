"""Contrast-maximisation loss for self-supervised M1 training (configs/evflow.yaml ``losses.cm``).

Per sample, with the window's events at normalised times tau in [0, 1]:

    L = - mean_{t_ref in reference_times} Var(G_s * IWE(flow, t_ref)) / Var(G_s * IWE(0, t_ref))
        + smoothness_weight * mean Charbonnier(forward differences of flow)       (px)
        + divergence_penalty * mean (du/dx + dv/dy)^2                             (px^2 / px^2)

IWE: bilinear splat on a canvas padded by ``pad_px`` (``models.cm_baseline.iwe_stack``, tested
equal to ``flow.iwe.iwe``), so events warped off the sensor still count (D-038). The loss is
applied to the finest prediction only.

Event-collapse guards (Shiba et al., Sensors 2022): (1) the multi-reference IWE: a flow field
that collapses the events onto few pixels at one reference time spreads them at the others;
(2) the divergence penalty: collapse needs a strongly contracting (negative-divergence) field,
while the scene flow of a translating or slowly rotating camera has small divergence.
``cm_diagnostics`` returns the collapse signatures the training-run-diagnostician looks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from evflow.models.cm_baseline import gauss_blur, iwe_stack


@dataclass(frozen=True)
class CMLossParams:
    reference_times: tuple[float, ...] = (0.0, 0.5, 1.0)
    smoothness_weight: float = 0.1
    divergence_penalty: float = 0.01
    blur_sigma: float = 1.0
    pad_px: int = 64
    max_events: int = 100_000
    charbonnier_eps: float = 1e-3

    @classmethod
    def from_config(cls, d: dict) -> CMLossParams:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "reference_times" in d:
            d["reference_times"] = tuple(float(v) for v in d["reference_times"])
        return cls(**d)


@dataclass
class WindowEvents:
    """Events of one training sample in the sample's (cropped, flipped) pixel frame."""

    x: torch.Tensor     # (N,) int64 or float
    y: torch.Tensor
    tau: torch.Tensor   # (N,) float, time normalised over the window, in [0, 1)


def subsample(ev: WindowEvents, max_events: int) -> WindowEvents:
    """Evenly spaced deterministic subsample (as M0b), so the cost per sample is bounded."""
    n = int(ev.tau.numel())
    if n <= max_events:
        return ev
    idx = torch.linspace(0, n - 1, max_events, device=ev.tau.device).round().long()
    return WindowEvents(ev.x[idx], ev.y[idx], ev.tau[idx])


def _contrast_ratio(ev: WindowEvents, flow: torch.Tensor, p: CMLossParams) -> torch.Tensor:
    h, w = flow.shape[-2:]
    a = dict(t_refs=p.reference_times, height=h, width=w, pad=p.pad_px)
    img = gauss_blur(iwe_stack(ev.x, ev.y, ev.tau, flow, **a), p.blur_sigma)
    with torch.no_grad():
        img0 = gauss_blur(iwe_stack(ev.x, ev.y, ev.tau, torch.zeros_like(flow), **a),
                          p.blur_sigma)
        v0 = img0.flatten(1).var(dim=1).clamp(min=1e-12)
    return (img.flatten(1).var(dim=1) / v0).mean()


def charbonnier_tv(flow: torch.Tensor, eps: float) -> torch.Tensor:
    dx = flow[..., :, 1:] - flow[..., :, :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    return torch.sqrt(dx * dx + eps * eps).mean() + torch.sqrt(dy * dy + eps * eps).mean()


def divergence(flow: torch.Tensor) -> torch.Tensor:
    """du/dx + dv/dy by forward differences, (..., H-1, W-1)."""
    du = flow[..., 0, :-1, 1:] - flow[..., 0, :-1, :-1]
    dv = flow[..., 1, 1:, :-1] - flow[..., 1, :-1, :-1]
    return du + dv


def cm_loss(flow: torch.Tensor, events: list[WindowEvents], p: CMLossParams
            ) -> tuple[torch.Tensor, dict[str, float]]:
    """Batch mean of the per-sample loss. ``flow`` (B, 2, H, W) float, finest prediction.
    Samples with fewer than 2 events have no data term (their regularisers still apply)."""
    if flow.ndim != 4 or flow.shape[1] != 2 or flow.shape[0] != len(events):
        raise ValueError(f"flow {tuple(flow.shape)} vs {len(events)} event sets")
    flow = flow.float()
    data_terms = []
    for i, ev in enumerate(events):
        if ev.tau.numel() < 2:
            continue
        ev = subsample(ev, p.max_events)
        data_terms.append(-_contrast_ratio(ev, flow[i], p))
    data = torch.stack(data_terms).mean() if data_terms else flow.sum() * 0
    smooth = charbonnier_tv(flow, p.charbonnier_eps)
    div = divergence(flow).pow(2).mean()
    loss = data + p.smoothness_weight * smooth + p.divergence_penalty * div
    parts = {"cm_data": float(data.detach()), "cm_smooth": float(smooth.detach()),
             "cm_div": float(div.detach())}
    return loss, parts


@torch.no_grad()
def cm_diagnostics(flow: torch.Tensor, ev: WindowEvents, *, t_ref: float = 1.0, pad_px: int = 64
                   ) -> dict[str, float]:
    """Collapse signatures of one window: mean |flow| at event pixels, fraction of non-empty IWE
    pixels relative to zero flow (rounded splat), IWE variance ratio, mean |divergence|."""
    h, w = flow.shape[-2:]
    out = {"flow_mag_events": float("nan"), "iwe_nonempty_ratio": float("nan"),
           "iwe_var_ratio": float("nan"),
           "abs_div": float(divergence(flow.float()).abs().mean())}
    if ev.tau.numel() < 2:
        return out
    a = dict(t_refs=(t_ref,), height=h, width=w, pad=pad_px)
    img = iwe_stack(ev.x, ev.y, ev.tau, flow.float(), **a)[0]
    img0 = iwe_stack(ev.x, ev.y, ev.tau, torch.zeros_like(flow, dtype=torch.float32), **a)[0]
    xi = ev.x.long() if not torch.is_floating_point(ev.x) else torch.round(ev.x).long()
    yi = ev.y.long() if not torch.is_floating_point(ev.y) else torch.round(ev.y).long()
    fe = flow.float()[:, yi.clamp(0, h - 1), xi.clamp(0, w - 1)]
    out["flow_mag_events"] = float(torch.linalg.vector_norm(fe, dim=0).mean())
    out["iwe_nonempty_ratio"] = float((img > 0.5).sum() / (img0 > 0.5).sum().clamp(min=1))
    out["iwe_var_ratio"] = float(img.var() / img0.var().clamp(min=1e-12))
    return out
