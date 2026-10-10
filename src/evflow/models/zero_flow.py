"""M0a: zero flow. Anchors every table: EPE = mean |GT| over the mask, FWL = RSAT = 1 exactly."""

from __future__ import annotations

import torch


def zero_flow(height: int, width: int, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Forward flow (2, H, W) of zeros, pixels over the window."""
    return torch.zeros((2, height, width), dtype=dtype, device=device)
