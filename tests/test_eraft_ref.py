"""Vendored E-RAFT (R1): builds, runs, warm-starts; checkpoints load (dataset marker)."""

from pathlib import Path

import pytest
import torch
import yaml

from evflow.models.eraft_ref.image_utils import ImagePadder, forward_interpolate_pytorch
from evflow.models.eraft_runner import WarmStartRunner, build_eraft, load_eraft

REPO = Path(__file__).resolve().parents[1]


def test_forward_shapes_and_warm_start():
    torch.manual_seed(0)
    model = build_eraft(15).eval()
    run = WarmStartRunner(model, iters=2)
    v0, v1, v2 = (torch.randn(15, 64, 96) for _ in range(3))
    f1 = run(v0, v1)
    assert f1.shape == (2, 64, 96)
    assert run.flow_init is not None and run.flow_init.shape == (1, 2, 8, 12)
    f2 = run(v1, v2)
    assert f2.shape == (2, 64, 96)
    run.reset()
    assert run.flow_init is None


def test_padder_left_top():
    p = ImagePadder(min_size=32)
    x = torch.zeros(1, 15, 260, 346)
    y = p.pad(x)
    assert y.shape[-2:] == (288, 352)
    assert p.unpad(y).shape == x.shape


def test_forward_interpolate_constant_flow():
    flow = torch.zeros(1, 2, 6, 7)
    flow[:, 0] = 1.0  # every pixel moves one column right
    out = forward_interpolate_pytorch(flow)
    assert torch.allclose(out[0, 0, :, 1:], torch.ones(6, 6))
    assert torch.all(out[0, 0, :, 0] == 0)  # nothing lands in the first column


@pytest.mark.dataset
@pytest.mark.parametrize("name", ["eraft_dsec", "eraft_mvsec20"])
def test_checkpoints_load_strict(name):
    cfg = yaml.safe_load((REPO / "configs/eraft.yaml").read_text())
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    model = load_eraft(paths["checkpoints"][name], sha256=cfg["checkpoints"][name]["sha256"])
    assert sum(p.numel() for p in model.parameters()) > 5_000_000
