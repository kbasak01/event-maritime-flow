"""Supervised multi-scale masked EPE."""

import pytest
import torch

from evflow.losses.supervised import masked_epe, multiscale_epe


def test_zero_at_gt_and_value():
    gt = torch.randn(2, 2, 8, 8)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    assert float(masked_epe(gt.clone(), gt, mask)) == 0.0
    pred = gt.clone()
    pred[:, 0] += 3.0
    pred[:, 1] += 4.0
    assert float(masked_epe(pred, gt, mask)) == pytest.approx(5.0, abs=1e-5)


def test_mask_respected_and_per_sample_mean():
    gt = torch.zeros(2, 2, 4, 4)
    pred = torch.zeros(2, 2, 4, 4)
    pred[0, 0, 0, 0] = 10.0     # masked out below
    pred[1, 0] = 2.0
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = False
    mask[1, 2:] = False         # sample 1: 8 px, each EPE 2
    assert float(masked_epe(pred, gt, mask)) == pytest.approx((0.0 + 2.0) / 2)


def test_empty_sample_excluded_and_all_empty_is_zero():
    gt = torch.zeros(2, 2, 4, 4)
    pred = torch.ones(2, 2, 4, 4, requires_grad=True)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[1] = True
    assert float(masked_epe(pred, gt, mask)) == pytest.approx(2 ** 0.5, rel=1e-5)
    loss = masked_epe(pred, gt, torch.zeros_like(mask))
    loss.backward()
    assert float(loss) == 0.0


def test_gradient_moves_towards_gt():
    gt = torch.full((1, 2, 4, 4), 2.0)
    pred = torch.zeros(1, 2, 4, 4, requires_grad=True)
    masked_epe(pred, gt, torch.ones(1, 4, 4, dtype=torch.bool)).backward()
    assert (pred.grad < 0).all()


def test_multiscale_upsamples_without_rescale():
    gt = torch.full((1, 2, 16, 16), 1.5)
    preds = [torch.full((1, 2, 2, 2), 1.5), torch.full((1, 2, 16, 16), 1.0)]
    loss, parts = multiscale_epe(preds, gt, torch.ones(1, 16, 16, dtype=torch.bool), [0.5, 1.0])
    assert parts[0] == pytest.approx(0.0, abs=1e-6)
    assert parts[1] == pytest.approx(0.5 * 2 ** 0.5, rel=1e-5)
    assert float(loss) == pytest.approx(0.5 * 2 ** 0.5, rel=1e-5)
