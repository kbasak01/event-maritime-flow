"""R1 inference: the vendored E-RAFT network with E-RAFT's warm start (``test.py``,
``TestRaftEventsWarm``): the low-resolution flow of the previous sample, forward-interpolated, is
the ``flow_init`` of the next one; the chain is reset when samples are not consecutive.

Input volumes are 15-bin voxel grids built by ``evflow.events.representations.voxel_grid`` with
``t_range = (t[0], t[-1])`` of the slice and ``normalize=True`` (E-RAFT's convention, D-021).
"""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path

import torch

from evflow.models.eraft_ref.eraft import ERAFT
from evflow.models.eraft_ref.image_utils import forward_interpolate_pytorch


def sha256_file(path: str | PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_eraft(num_bins: int = 15) -> ERAFT:
    return ERAFT({"subtype": "warm_start"}, n_first_channels=num_bins)


def load_eraft(checkpoint: str | PathLike, *, sha256: str, num_bins: int = 15,
               device: str | torch.device = "cpu") -> ERAFT:
    """Build E-RAFT and load ``checkpoint['model']``. The file is pickled with numpy objects, so
    it is only unpickled after its SHA-256 matches the pinned value (configs/eraft.yaml)."""
    got = sha256_file(checkpoint)
    if got != sha256:
        raise ValueError(f"{checkpoint}: sha256 {got} != pinned {sha256}")
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model = build_eraft(num_bins)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


class WarmStartRunner:
    """Runs E-RAFT over a chain of samples; ``reset()`` starts a new chain."""

    def __init__(self, model: ERAFT, *, iters: int = 12) -> None:
        self.model = model
        self.iters = iters
        self.flow_init: torch.Tensor | None = None

    def reset(self) -> None:
        self.flow_init = None

    @torch.no_grad()
    def __call__(self, vox_old: torch.Tensor, vox_new: torch.Tensor) -> torch.Tensor:
        """(B, H, W) voxel grids of the previous and the current window -> flow (2, H, W)."""
        dev = next(self.model.parameters()).device
        flow_low, preds = self.model(vox_old[None].to(dev), vox_new[None].to(dev),
                                     iters=self.iters, flow_init=self.flow_init)
        self.flow_init = forward_interpolate_pytorch(flow_low)
        return preds[-1][0]
