"""M1 network: size, output scales and padding."""

import pytest
import torch

from evflow.models.evflownet import N_SCALES, EVFlowNet, n_params, pad_to_multiple, unpad


def test_param_count_in_plan_range():
    # IMPLEMENTATION_PLAN §1.4: "EV-FlowNet-style U-Net (≈ 5–15 M params)"
    n = n_params(EVFlowNet(in_channels=5, base_channels=32))
    assert 5e6 <= n <= 15e6, n


@pytest.mark.parametrize("cin", [2, 5, 15])
def test_output_scales(cin):
    net = EVFlowNet(in_channels=cin, base_channels=8).eval()
    flows = net(torch.randn(2, cin, 64, 96))
    assert len(flows) == N_SCALES
    for k, f in zip(range(3, -1, -1), flows, strict=True):
        assert f.shape == (2, 2, 64 >> k, 96 >> k)


def test_rejects_non_multiple_of_16():
    with pytest.raises(ValueError):
        EVFlowNet(in_channels=5, base_channels=8)(torch.randn(1, 5, 260, 346))


def test_pad_roundtrip():
    x = torch.randn(1, 5, 260, 346)
    xp, hw = pad_to_multiple(x, 16)
    assert xp.shape[-2:] == (272, 352)
    assert torch.equal(xp[..., 260:, :], torch.zeros_like(xp[..., 260:, :]))
    assert torch.equal(unpad(xp, hw), x)


def test_gradients_reach_input_layer():
    net = EVFlowNet(in_channels=5, base_channels=8)
    loss = sum(f.abs().mean() for f in net(torch.randn(2, 5, 32, 32)))
    loss.backward()
    assert net.enc[0][0].weight.grad is not None
    assert net.enc[0][0].weight.grad.abs().sum() > 0
