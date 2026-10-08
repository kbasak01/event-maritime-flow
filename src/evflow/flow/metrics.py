"""Torch flow metrics; same definitions as ``metrics_ref.py`` (tested equal to it).

Flow tensors are (2, H, W), pixels over the window, forward. ``valid_gt`` is required; the other
masks are optional and combined with AND. See ``metrics_ref`` for the exact definitions.
"""

from __future__ import annotations

import math

import torch

from evflow.flow.metrics_ref import NPE_THRESHOLDS, OUTLIER_ABS_PX, OUTLIER_REL


def eval_mask(*, valid_gt, events_in_window=None, crop=None, class_mask=None) -> torch.Tensor:
    mask = valid_gt.bool().clone()
    for m in (events_in_window, crop, class_mask):
        if m is not None:
            if m.shape != mask.shape:
                raise ValueError(f"mask shape {tuple(m.shape)} != valid_gt {tuple(mask.shape)}")
            mask &= m.bool()
    return mask


def epe_map(flow_pred: torch.Tensor, flow_gt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(flow_pred - flow_gt, dim=-3)


def ae_map(flow_pred: torch.Tensor, flow_gt: torch.Tensor) -> torch.Tensor:
    u, v = flow_pred[..., 0, :, :], flow_pred[..., 1, :, :]
    ug, vg = flow_gt[..., 0, :, :], flow_gt[..., 1, :, :]
    c = (1.0 + u * ug + v * vg) / torch.sqrt((1.0 + u * u + v * v) * (1.0 + ug * ug + vg * vg))
    return torch.rad2deg(torch.arccos(torch.clamp(c, -1.0, 1.0)))


@torch.no_grad()
def flow_metrics(
    flow_pred: torch.Tensor, flow_gt: torch.Tensor, *, valid_gt, events_in_window=None, crop=None,
    class_mask=None,
) -> dict[str, float]:
    if flow_pred.shape != flow_gt.shape or flow_pred.ndim != 3 or flow_pred.shape[0] != 2:
        raise ValueError(f"flow shapes must both be (2, H, W): {flow_pred.shape}, {flow_gt.shape}")
    if valid_gt.shape != flow_gt.shape[1:]:
        raise ValueError(f"valid_gt shape {tuple(valid_gt.shape)} != {tuple(flow_gt.shape[1:])}")
    mask = eval_mask(valid_gt=valid_gt, events_in_window=events_in_window, crop=crop,
                     class_mask=class_mask)
    # Reported metrics are computed in float64: arccos near 1 loses ~1e-3 deg in float32.
    flow_pred, flow_gt = flow_pred.double(), flow_gt.double()
    n = int(mask.sum())
    keys = ["epe", "outlier_pct", *(f"npe_{k}" for k in NPE_THRESHOLDS), "ae_deg"]
    if n == 0:
        return {**{k: math.nan for k in keys}, "n_px": 0}
    e = epe_map(flow_pred, flow_gt)[mask]
    gt_mag = torch.linalg.vector_norm(flow_gt, dim=0)[mask]
    out = {
        "epe": float(e.mean()),
        "outlier_pct": float(100.0 * ((e > OUTLIER_ABS_PX) & (e > OUTLIER_REL * gt_mag))
                             .to(e.dtype).mean()),
    }
    for k in NPE_THRESHOLDS:
        out[f"npe_{k}"] = float(100.0 * (e > k).to(e.dtype).mean())
    out["ae_deg"] = float(ae_map(flow_pred, flow_gt)[mask].mean())
    out["n_px"] = n
    return out
