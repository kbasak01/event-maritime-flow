"""M1: EV-FlowNet-style U-Net (Zhu et al., RSS 2018 architecture idea; own implementation).

Input: one event representation (C, H, W) of one window (H, W multiples of 16, see
``pad_to_multiple``). Output: forward flow in pixels over the window (docs/CONVENTIONS.md), one
prediction per decoder scale, coarse to fine, every one in full-resolution pixel units (a coarse
head is upsampled bilinearly without rescaling its values). Encoder: four stride-2 3x3 convs
(2b, 4b, 8b, 16b channels); two residual blocks at 1/16; decoder: bilinear x2 upsampling, concat
of the encoder skip and the previous (upsampled) flow, 3x3 conv; a 1x1 flow head per scale.
Only Conv/BatchNorm/ReLU/Resize/Concat: ONNX-exportable (Phase 8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SCALES = 4


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


def _up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)


class EVFlowNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, n_res: int = 2) -> None:
        super().__init__()
        b = base_channels
        enc = [2 * b, 4 * b, 8 * b, 16 * b]
        self.enc = nn.ModuleList()
        cin = in_channels
        for c in enc:
            self.enc.append(_conv(cin, c, stride=2))
            cin = c
        self.res = nn.Sequential(*[_ResBlock(enc[-1]) for _ in range(n_res)])
        # decoder stage s goes from 1/2^(4-s) to 1/2^(3-s); skips are enc[2], enc[1], enc[0], none
        dec = [8 * b, 4 * b, 2 * b, b]
        skips = [enc[2], enc[1], enc[0], 0]
        self.dec = nn.ModuleList()
        self.heads = nn.ModuleList()
        cin = enc[-1]
        for s, (c, sk) in enumerate(zip(dec, skips, strict=True)):
            self.dec.append(_conv(cin + sk + (2 if s else 0), c))
            self.heads.append(nn.Conv2d(c, 2, 1))
            cin = c

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """List of N_SCALES flows (B, 2, H/2^k, W/2^k), k = 3..0 (coarse to fine)."""
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(f"input H, W must be multiples of 16, got {tuple(x.shape[-2:])}")
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
        x = self.res(x)
        flows: list[torch.Tensor] = []
        for s, (layer, head) in enumerate(zip(self.dec, self.heads, strict=True)):
            parts = [_up(x)]
            if s < 3:
                parts.append(skips[2 - s])
            if flows:
                parts.append(_up(flows[-1]))
            x = layer(torch.cat(parts, dim=1))
            flows.append(head(x))
        return flows


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    """Zero-pad (..., H, W) at the bottom/right to multiples of ``multiple``; returns the
    original (H, W) for ``unpad``."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    return (F.pad(x, (0, pw, 0, ph)) if ph or pw else x), (h, w)


def unpad(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return x[..., : hw[0], : hw[1]]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
