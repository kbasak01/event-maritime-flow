"""Flow metrics: hand-computed 4x4 cases, explicit masks, torch == numpy (§1.6, checklist C1)."""

import math

import numpy as np
import pytest
import torch

from evflow.flow import metrics as tm
from evflow.flow import metrics_ref as ref


def _case():
    """4x4 field. Flow layout (2, H, W): channel 0 = u, 1 = v."""
    gt = np.ones((2, 4, 4))
    pred = np.ones((2, 4, 4))
    valid = np.ones((4, 4), bool)
    gt[:, 0, 0] = (3, 4)  # A: EPE 0
    pred[:, 0, 0] = (3, 4)
    gt[:, 0, 1] = (0, 0)  # B: invalid GT, must be ignored
    pred[:, 0, 1] = (50, 50)
    valid[0, 1] = False
    gt[:, 1, 1] = (100, 0)  # C: EPE 4 > 3 but < 5 % of 100
    pred[:, 1, 1] = (104, 0)
    gt[:, 2, 2] = (1, 0)  # D: EPE 4, outlier
    pred[:, 2, 2] = (5, 0)
    gt[:, 3, 3] = (0, 2)  # E: EPE 1.5
    pred[:, 3, 3] = (0, 3.5)
    gt[:, 3, 0] = (1, 1)  # F: EPE exactly 1 (npe strict >)
    pred[:, 3, 0] = (2, 1)
    return pred, gt, valid


def _ae(u, v, ug, vg):
    c = (1 + u * ug + v * vg) / math.sqrt((1 + u * u + v * v) * (1 + ug * ug + vg * vg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def test_hand_case_all_metrics():
    pred, gt, valid = _case()
    m = ref.flow_metrics(pred, gt, valid_gt=valid)
    n = 15
    assert m["n_px"] == n
    assert m["epe"] == pytest.approx((0 + 4 + 4 + 1.5 + 1) / n, abs=1e-12)
    assert m["outlier_pct"] == pytest.approx(100 * 1 / n)  # only D; C is < 5 % of |GT|
    assert m["npe_1"] == pytest.approx(100 * 3 / n)        # C, D, E (F == 1 not counted)
    assert m["npe_2"] == pytest.approx(100 * 2 / n)
    assert m["npe_3"] == pytest.approx(100 * 2 / n)
    ae = _ae(104, 0, 100, 0) + _ae(5, 0, 1, 0) + _ae(0, 3.5, 0, 2) + _ae(2, 1, 1, 1)
    assert m["ae_deg"] == pytest.approx(ae / n, abs=1e-9)


def test_outlier_needs_both_conditions():
    gt = np.zeros((2, 1, 3))
    pred = np.zeros((2, 1, 3))
    valid = np.ones((1, 3), bool)
    gt[0, 0] = [100.0, 1.0, 0.5]
    pred[0, 0] = [104.0, 3.5, 4.0]   # EPE 4 (<5%), 2.5 (<3 px), 3.5 (>3 and >5%)
    m = ref.flow_metrics(pred, gt, valid_gt=valid)
    assert m["outlier_pct"] == pytest.approx(100 / 3)


def test_masks_are_anded():
    pred, gt, valid = _case()
    ev = np.ones((4, 4), bool)  # drop D
    ev[2, 2] = False
    crop = ref.crop_mask(4, 4, rows=(0, 3), cols=(0, 4))     # drop row 3 (E, F + 2 others)
    m = ref.flow_metrics(pred, gt, valid_gt=valid, events_in_window=ev, crop=crop)
    assert m["n_px"] == 15 - 1 - 4
    assert m["outlier_pct"] == 0.0
    assert m["epe"] == pytest.approx(4 / 10)
    cls = np.zeros((4, 4), bool)
    cls[1, 1] = True
    m = ref.flow_metrics(pred, gt, valid_gt=valid, class_mask=cls)
    assert m["n_px"] == 1 and m["epe"] == pytest.approx(4.0)


def test_valid_gt_is_required_and_never_derived():
    pred, gt, _ = _case()
    with pytest.raises(TypeError):
        ref.flow_metrics(pred, gt)  # no default mask
    with pytest.raises(TypeError):
        tm.flow_metrics(torch.from_numpy(pred), torch.from_numpy(gt))
    # Pixel B has GT (0, 0) but is declared valid here -> it is counted (no hidden GT!=0 rule).
    m = ref.flow_metrics(pred, gt, valid_gt=np.ones((4, 4), bool))
    assert m["n_px"] == 16


def test_empty_mask_gives_nan():
    pred, gt, _ = _case()
    m = ref.flow_metrics(pred, gt, valid_gt=np.zeros((4, 4), bool))
    assert m["n_px"] == 0 and math.isnan(m["epe"]) and math.isnan(m["outlier_pct"])


def test_mask_shape_checked():
    pred, gt, valid = _case()
    with pytest.raises(ValueError):
        ref.flow_metrics(pred, gt, valid_gt=valid[:3])
    with pytest.raises(ValueError):
        ref.flow_metrics(pred[:, :3], gt, valid_gt=valid)


def test_epe_map_units_pixels():
    pred, gt, _ = _case()
    e = ref.epe_map(pred, gt)
    assert e.shape == (4, 4) and e[1, 1] == pytest.approx(4.0) and e[3, 3] == pytest.approx(1.5)


@pytest.mark.parametrize("seed", range(5))
def test_torch_equals_numpy(seed):
    rng = np.random.default_rng(seed)
    gt = rng.normal(0, 6, (2, 20, 24))
    pred = gt + rng.normal(0, 3, gt.shape)
    valid = rng.random((20, 24)) > 0.2
    ev = rng.random((20, 24)) > 0.3
    crop = ref.crop_mask(20, 24, rows=(2, 18), cols=(1, 23))
    a = ref.flow_metrics(pred, gt, valid_gt=valid, events_in_window=ev, crop=crop)
    t = lambda z: torch.from_numpy(z)  # noqa: E731
    b = tm.flow_metrics(t(pred), t(gt), valid_gt=t(valid), events_in_window=t(ev), crop=t(crop))
    assert a.keys() == b.keys()
    for k in a:
        assert b[k] == pytest.approx(a[k], abs=1e-6), k
    np.testing.assert_allclose(tm.epe_map(t(pred), t(gt)).numpy(), ref.epe_map(pred, gt),
                               atol=1e-6)


def test_torch_float32_input_matches_float64_reference():
    rng = np.random.default_rng(11)
    gt = rng.normal(0, 20, (2, 64, 64))
    pred = gt + rng.normal(0, 0.3, gt.shape)
    valid = np.ones((64, 64), bool)
    a = ref.flow_metrics(pred, gt, valid_gt=valid)
    b = tm.flow_metrics(torch.from_numpy(pred).float(), torch.from_numpy(gt).float(),
                        valid_gt=torch.from_numpy(valid))
    assert b["ae_deg"] == pytest.approx(a["ae_deg"], rel=1e-5)
    assert b["epe"] == pytest.approx(a["epe"], rel=1e-5)


def test_epe_map_is_the_reference_formula():
    """epe_map = sqrt(du^2 + dv^2) as metrics_ref (float64, same operations; equal to a few ulp,
    the kernels round differently). Replaces torch.linalg.vector_norm over the 2-channel dim,
    which took ~36 ms per MVSEC frame on CPU (Phase 6, D-052)."""
    rng = np.random.default_rng(3)
    gt = rng.normal(0, 6, (2, 260, 346))
    pred = gt + rng.normal(0, 3, gt.shape)
    t = lambda z: torch.from_numpy(z)  # noqa: E731
    np.testing.assert_allclose(tm.epe_map(t(pred), t(gt)).numpy(), ref.epe_map(pred, gt),
                               rtol=1e-14, atol=0)
