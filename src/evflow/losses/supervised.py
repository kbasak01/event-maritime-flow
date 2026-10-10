"""Supervised loss for M1: multi-scale masked EPE against GT flow (configs/evflow.yaml
``losses.supervised``).

Every prediction scale is in full-resolution pixel units (``models.evflownet``), so a coarse
prediction is upsampled bilinearly to the GT resolution without rescaling its values. Per sample:
mean L2 end-point error over the masked pixels; samples with no masked pixel contribute 0 and are
excluded from the batch mean. Scale weights are given coarse to fine.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_epe(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Batch mean of the per-sample masked mean EPE. ``pred``/``gt`` (B, 2, H, W), ``mask``
    (B, H, W) bool."""
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[1] != 2:
        raise ValueError(f"pred/gt must both be (B, 2, H, W): {pred.shape}, {gt.shape}")
    if mask.shape != (gt.shape[0], *gt.shape[2:]):
        raise ValueError(f"mask {tuple(mask.shape)} does not match gt {tuple(gt.shape)}")
    e = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)
    m = mask.to(e.dtype)
    n = m.flatten(1).sum(1)
    per = (e * m).flatten(1).sum(1) / n.clamp(min=1)
    has = n > 0
    if not bool(has.any()):
        return (pred * 0).sum()   # keeps the graph; no supervision in this batch
    return per[has].mean()


def multiscale_epe(preds: list[torch.Tensor], gt: torch.Tensor, mask: torch.Tensor,
                   weights: list[float]) -> tuple[torch.Tensor, list[float]]:
    """Weighted sum over scales of ``masked_epe`` of each prediction upsampled to the GT size.
    Returns the loss and the per-scale EPE values (floats, for logging)."""
    if len(preds) != len(weights):
        raise ValueError(f"{len(preds)} predictions but {len(weights)} weights")
    total, parts = 0.0, []
    for p, w in zip(preds, weights, strict=True):
        if p.shape[-2:] != gt.shape[-2:]:
            p = F.interpolate(p, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        e = masked_epe(p, gt, mask)
        total = total + w * e
        parts.append(float(e.detach()))
    return total, parts
