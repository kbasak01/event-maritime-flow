"""Self-supervised motion-compensation losses for M1 (configs/evflow.yaml ``losses.cm``).

Two objectives over the window's events at normalised times tau in [0, 1], warped by the
predicted flow to a reference time (bilinear splat on a canvas padded by ``pad_px``, so events
warped off the sensor still count, D-038):

``average_timestamp`` (default, D-047; own implementation after Zhu et al., CVPR 2019 and
Hagenaars et al., NeurIPS 2021): per polarity, the timestamp-weighted IWE divided by the count
IWE (average-timestamp image); its squares summed over pixels and divided by the number of
pixels hit by either polarity. Evaluated at t_ref = 1 with weights tau and at t_ref = 0 with
weights 1 - tau, and the two terms added (multi-reference).

``iwe_variance`` (D-043, superseded): - mean over ``reference_times`` of
Var(G_s * IWE(flow)) / Var(G_s * IWE(0)). It collapsed on real MVSEC (D-047) and is kept only to
reproduce that finding.

Both add ``smoothness_weight`` x mean Charbonnier(forward differences of flow) and
``divergence_penalty`` x mean (du/dx + dv/dy)^2, and are applied to the finest prediction only.

Event-collapse guards (Shiba et al., Sensors 2022): reference times at both ends of the window,
and the divergence penalty (collapse needs a strongly contracting field). ``cm_diagnostics``
returns the collapse signatures the training-run-diagnostician looks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from evflow.models.cm_baseline import gauss_blur, iwe_stack

OBJECTIVES = ("average_timestamp", "iwe_variance")


@dataclass(frozen=True)
class CMLossParams:
    objective: str = "average_timestamp"
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

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")


@dataclass
class WindowEvents:
    """Events of one training sample in the sample's (cropped, flipped) pixel frame."""

    x: torch.Tensor     # (N,) int64 or float
    y: torch.Tensor
    tau: torch.Tensor   # (N,) float, time normalised over the window, in [0, 1)
    p: torch.Tensor | None = None   # (N,) polarity {0, 1}; required by average_timestamp


def subsample(ev: WindowEvents, max_events: int) -> WindowEvents:
    """Evenly spaced deterministic subsample (as M0b), so the cost per sample is bounded."""
    n = int(ev.tau.numel())
    if n <= max_events:
        return ev
    idx = torch.linspace(0, n - 1, max_events, device=ev.tau.device).round().long()
    return WindowEvents(ev.x[idx], ev.y[idx], ev.tau[idx], None if ev.p is None else ev.p[idx])


def _contrast_ratio(ev: WindowEvents, flow: torch.Tensor, p: CMLossParams) -> torch.Tensor:
    h, w = flow.shape[-2:]
    a = dict(t_refs=p.reference_times, height=h, width=w, pad=p.pad_px)
    img = gauss_blur(iwe_stack(ev.x, ev.y, ev.tau, flow, **a), p.blur_sigma)
    with torch.no_grad():
        img0 = gauss_blur(iwe_stack(ev.x, ev.y, ev.tau, torch.zeros_like(flow), **a),
                          p.blur_sigma)
        v0 = img0.flatten(1).var(dim=1).clamp(min=1e-12)
    return (img.flatten(1).var(dim=1) / v0).mean()


def average_timestamp_images(ev: WindowEvents, flow: torch.Tensor, *, t_ref: float,
                             weights: torch.Tensor, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Count and weight-sum IWEs per polarity, (2, H + 2 pad, W + 2 pad) each: events warped to
    ``t_ref`` by the flow at their own (rounded) pixel, bilinear splat, channel = polarity."""
    h, w = flow.shape[-2:]
    hc, wc = h + 2 * pad, w + 2 * pad
    xi = ev.x.long() if not torch.is_floating_point(ev.x) else torch.round(ev.x.double()).long()
    yi = ev.y.long() if not torch.is_floating_point(ev.y) else torch.round(ev.y.double()).long()
    fe = flow.reshape(2, -1)[:, yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)]
    d = t_ref - ev.tau.to(flow.dtype)
    xw = ev.x.to(flow.dtype) + d * fe[0] + pad
    yw = ev.y.to(flow.dtype) + d * fe[1] + pad
    x0, y0 = torch.floor(xw), torch.floor(yw)
    fx, fy = xw - x0, yw - y0
    base = ev.p.long().clamp(0, 1) * (hc * wc)
    cnt = torch.zeros(2 * hc * wc, dtype=flow.dtype, device=flow.device)
    wts = torch.zeros_like(cnt)
    wv = weights.to(flow.dtype)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xt, yt = x0 + dx, y0 + dy
        k = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
        ok = (xt >= 0) & (xt < wc) & (yt >= 0) & (yt < hc)
        idx = base + (yt.clamp(0, hc - 1) * wc + xt.clamp(0, wc - 1)).long()
        k = k * ok
        cnt = cnt.scatter_add(0, idx, k)
        wts = wts.scatter_add(0, idx, k * wv)
    return cnt.view(2, hc, wc), wts.view(2, hc, wc)


def average_timestamp_loss(ev: WindowEvents, flow: torch.Tensor, pad: int) -> torch.Tensor:
    """Sum over t_ref in {1 (weights tau), 0 (weights 1 - tau)} of
    sum_pixels,polarity (weight IWE / count IWE)^2 / #pixels hit by either polarity."""
    if ev.p is None:
        raise ValueError("average_timestamp needs event polarities")
    tau = ev.tau.to(flow.dtype)
    total = flow.new_zeros(())
    for t_ref, wts in ((1.0, tau), (0.0, 1.0 - tau)):
        cnt, tsum = average_timestamp_images(ev, flow, t_ref=t_ref, weights=wts, pad=pad)
        avg = tsum / (cnt + 1e-9)
        hit = (cnt.sum(0) > 0).sum().clamp(min=1)
        total = total + (avg * avg).sum() / hit
    return total


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
        if p.objective == "average_timestamp":
            data_terms.append(average_timestamp_loss(ev, flow[i], p.pad_px))
        else:
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
