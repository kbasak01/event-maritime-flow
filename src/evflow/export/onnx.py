"""ONNX export of M1 and the parity statistics (Phase 8 steps 1-2; D-063).

The graph is the network only: input ``events`` (1, C, H, W), the event representation of one
window zero-padded to a multiple of 16 (``models.evflownet.pad_to_multiple``); output ``flow``
(1, 2, H, W), the finest of M1's four predictions, forward flow in pixels over the window
(docs/CONVENTIONS.md). The representation build (scatter-add) stays outside the graph and is
benchmarked separately in torch-CUDA (plan Phase 8 step 1). Shapes are static (no dynamic axes):
one file per input resolution, which is also what TensorRT builds fastest. Opset 17, legacy
TorchScript exporter (``dynamo=False``), constant folding on (BatchNorm folds into the convs in
eval mode). fp16 = the network cast to half, fp16 input and output.

A sidecar ``<file>.json`` records the run, checkpoint SHA-256, ONNX SHA-256, opset, shapes,
dtypes and the exporter versions; the auditor checks the checkpoint SHA against the run's
``selection.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evflow.models.evflownet import EVFlowNet

OPSET = 17
INPUT_NAME = "events"
OUTPUT_NAME = "flow"


class FinestFlow(nn.Module):
    """M1 with only its finest-scale flow as output (the one every evaluation uses)."""

    def __init__(self, net: EVFlowNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.net(events)[-1]


def padded_hw(height: int, width: int, multiple: int = 16) -> tuple[int, int]:
    """(H, W) after ``pad_to_multiple``: 260x346 -> 272x352."""
    return height + (-height) % multiple, width + (-width) % multiple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def export_m1(model: EVFlowNet, path: Path, *, in_channels: int, height: int, width: int,
              fp16: bool, opset: int = OPSET, meta: dict | None = None) -> dict:
    """Export ``model`` (unchanged; a copy is converted) to ``path`` for a fixed (1, C, H, W)
    input; H, W must already be multiples of 16. Returns the sidecar dict (also written)."""
    import onnx

    if height % 16 or width % 16:
        raise ValueError(f"export shape must be padded to /16, got {height}x{width}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dtype = torch.float16 if fp16 else torch.float32
    net = FinestFlow(copy.deepcopy(model)).to(device=device, dtype=dtype).eval()
    gen = torch.Generator().manual_seed(0)
    dummy = torch.randn(1, in_channels, height, width, generator=gen).to(device=device,
                                                                        dtype=dtype)
    with torch.no_grad():
        torch.onnx.export(net, (dummy,), str(path), dynamo=False, opset_version=opset,
                          input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          do_constant_folding=True, dynamic_axes=None)
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    got = {o.domain: o.version for o in m.opset_import}.get("", None)
    if got != opset:
        raise RuntimeError(f"{path}: exported opset {got}, expected {opset}")
    info = {
        **(meta or {}),
        "onnx": path.name,
        "onnx_sha256": sha256_file(path),
        "opset": opset,
        "precision": "fp16" if fp16 else "fp32",
        "input": {"name": INPUT_NAME, "shape": [1, in_channels, height, width],
                  "dtype": "float16" if fp16 else "float32"},
        "output": {"name": OUTPUT_NAME, "shape": [1, 2, height, width],
                   "dtype": "float16" if fp16 else "float32",
                   "meaning": "finest M1 flow, px over the window, forward"},
        "dynamic_axes": None,
        "exporter": "torch.onnx.export (TorchScript, dynamo=False), do_constant_folding=True",
        "export_device": str(device),
        "representation_build": "outside the graph (torch-CUDA, benchmarked separately)",
        "n_nodes": len(m.graph.node),
        "op_types": sorted({n.op_type for n in m.graph.node}),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__},
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def parity_stats(ref: np.ndarray, test: np.ndarray, *, mask: np.ndarray) -> dict:
    """Flow differences between a reference and a test prediction, both (N, 2, H, W) px;
    ``mask`` (N, H, W) is the evaluation mask. ``max_px`` is the largest vector difference
    |test - ref| over every pixel of every window (the gated number), ``*_masked`` the same over
    the evaluation mask; ``max_component_px`` the largest |du| or |dv|. A non-finite test value
    makes every max infinite and is counted."""
    ref = np.asarray(ref, np.float64)
    test = np.asarray(test, np.float64)
    if ref.shape != test.shape or ref.ndim != 4 or ref.shape[1] != 2:
        raise ValueError(f"shapes must both be (N, 2, H, W): {ref.shape}, {test.shape}")
    mask = np.asarray(mask, bool)
    if mask.shape != (ref.shape[0], *ref.shape[2:]):
        raise ValueError(f"mask shape {mask.shape} != {(ref.shape[0], *ref.shape[2:])}")
    d = test - ref
    bad = ~np.isfinite(d).all(axis=1)
    mag = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    mag[bad] = np.inf
    comp = np.abs(d).max(axis=1)
    comp[bad] = np.inf
    mm = mag[mask]
    return {
        "max_px": float(mag.max()),
        "max_component_px": float(comp.max()),
        "mean_px": float(mag[~bad].mean()) if (~bad).any() else float("inf"),
        "p99_px": float(np.percentile(mag[~bad], 99)) if (~bad).any() else float("inf"),
        "max_px_masked": float(mm.max()) if mm.size else float("nan"),
        "mean_px_masked": float(mm[np.isfinite(mm)].mean()) if np.isfinite(mm).any()
        else float("nan"),
        "n_px": int(mag.size),
        "n_px_masked": int(mm.size),
        "n_non_finite": int(bad.sum()),
        "per_window_max_px": [float(v) for v in mag.reshape(mag.shape[0], -1).max(axis=1)],
    }
