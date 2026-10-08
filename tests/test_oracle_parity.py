"""Parity with the vendored MIT tudelft/event_flow metrics (checklist C1/C2, plan Phase 1 step 6).

Mapping between conventions (ours -> oracle):
- event_list rows are (ts, y, x, p) with ts = our normalised time tau in [0, 1];
- flow_scaling = 1 and dt_gt = dt_input = 1, so oracle flow = our flow in px over the window;
- oracle FWL / RSAT warp to tref = max_ts = 1 with rounding -> ours with t_ref=1, 'round';
- oracle AEE masks: event_mask AND GT != (0, 0) -> our events_in_window AND valid_gt built by the
  caller from the same rule.
Known definitional difference, kept on purpose: the oracle's %Out compares the EPE with 5 % of the
*predicted* flow magnitude; ours (MVSEC/KITTI convention, docs/CONVENTIONS.md) uses |GT|. Parity of
%Out is therefore tested on fields with |pred| == |GT| per pixel, and the divergence is pinned by
``test_outlier_reference_magnitude_differs``.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from evflow.flow import fwl as tfwl
from evflow.flow import iwe_ref, metrics_ref

from oracles import event_flow_metrics as ef

REPO = Path(__file__).resolve().parents[1]
H, W = 24, 30
T_RANGE = (10_000, 30_000)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # the oracle allocates with the default dtype
    yield
    torch.set_default_dtype(old)


def _events(seed, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, W, n).astype(np.uint16)
    y = rng.integers(0, H, n).astype(np.uint16)
    t = np.sort(rng.integers(*T_RANGE, n)).astype(np.int64)
    p = rng.integers(0, 2, n).astype(np.int8)
    flow = rng.normal(0, 3, (2, H, W))
    return x, y, t, p, flow


def _oracle_inputs(x, y, t, p, *, event_mask=None, gt=None):
    tau = iwe_ref.normalize_time(t, T_RANGE)
    ev = torch.from_numpy(np.stack([tau, y, x, p], axis=1).astype(np.float64))[None]
    pol = torch.from_numpy(np.stack([p == 1, p == 0], axis=1).astype(np.float64))[None]
    em = np.ones((H, W)) if event_mask is None else event_mask.astype(np.float64)
    inputs = {
        "event_list": ev,
        "event_list_pol_mask": pol,
        "event_mask": torch.from_numpy(em)[None, None],
        "dt_input": torch.ones(1),
        "dt_gt": torch.ones(1),
    }
    if gt is not None:
        inputs["gtflow"] = torch.from_numpy(gt)[None]
    return inputs


def _oracle(cls, flow, inputs):
    m = cls({"loader": {"resolution": [H, W]}, "loss": {}}, "cpu", flow_scaling=1)
    m.event_flow_association([torch.from_numpy(flow)[None]], inputs)
    return m.forward()


@pytest.mark.parametrize("seed", range(4))
def test_fwl_parity(seed):
    x, y, t, p, flow = _events(seed)
    ours = iwe_ref.fwl(x, y, t, flow, t_range=T_RANGE, t_ref=1.0, height=H, width=W,
                       interpolation="round")
    ours_t = tfwl.fwl(*(torch.from_numpy(a.astype(np.int64)) for a in (x, y, t)),
                      torch.from_numpy(flow), t_range=T_RANGE, t_ref=1.0, height=H, width=W,
                      interpolation="round")
    oracle = float(_oracle(ef.FWL, flow, _oracle_inputs(x, y, t, p))[0])
    assert ours == pytest.approx(oracle, abs=1e-6)
    assert ours_t == pytest.approx(oracle, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_rsat_parity(seed):
    x, y, t, p, flow = _events(seed)
    ours = iwe_ref.rsat(x, y, t, p, flow, t_range=T_RANGE, t_ref=1.0, height=H, width=W,
                        interpolation="round")
    oracle = float(_oracle(ef.RSAT, flow, _oracle_inputs(x, y, t, p))[0])
    assert ours == pytest.approx(oracle, abs=1e-6)


@pytest.mark.parametrize("round_idx", [True, False])
@pytest.mark.parametrize("t_ref", [0.0, 0.5, 1.0])
def test_iwe_parity(round_idx, t_ref):
    x, y, t, p, flow = _events(7)
    tau = iwe_ref.normalize_time(t, T_RANGE)
    ev = torch.from_numpy(np.stack([tau, y, x, p], axis=1).astype(np.float64))[None]
    ev_flow = torch.from_numpy(np.stack([flow[1, y, x], flow[0, y, x]], axis=1))[None]  # (y, x)
    idx, wts = ef.get_interpolation(ev, ev_flow, t_ref, [H, W], 1, round_idx=round_idx)
    oracle = ef.interpolate(idx.long(), wts, [H, W])[0, 0].numpy()
    ours = iwe_ref.iwe(x, y, t, flow, t_range=T_RANGE, t_ref=t_ref, height=H, width=W,
                       interpolation="round" if round_idx else "bilinear")
    np.testing.assert_allclose(ours, oracle, rtol=0, atol=1e-6)


def _gt_and_event_mask(seed):
    rng = np.random.default_rng(seed)
    gt = rng.normal(0, 8, (2, H, W))
    gt[:, rng.random((H, W)) < 0.15] = 0.0  # invalid GT pixels (oracle rule: both components 0)
    event_mask = rng.random((H, W)) > 0.3
    return rng, gt, event_mask


@pytest.mark.parametrize("seed", range(4))
def test_aee_and_outlier_parity(seed):
    rng, gt, event_mask = _gt_and_event_mask(seed)
    # |pred| == |GT| per pixel (rotation), so the oracle's |pred|-based %Out equals our |GT| one.
    ang = rng.uniform(-1.2, 1.2, (H, W))
    c, s = np.cos(ang), np.sin(ang)
    pred = np.stack([c * gt[0] - s * gt[1], s * gt[0] + c * gt[1]])
    x, y, t, p, _ = _events(seed, n=50)
    aee, frac_out = _oracle(ef.AEE, pred, _oracle_inputs(x, y, t, p, event_mask=event_mask, gt=gt))
    valid_gt = ~((gt[0] == 0) & (gt[1] == 0))
    ours = metrics_ref.flow_metrics(pred, gt, valid_gt=valid_gt, events_in_window=event_mask)
    assert ours["epe"] == pytest.approx(float(aee[0]), abs=1e-6)
    assert ours["outlier_pct"] == pytest.approx(100.0 * float(frac_out), abs=1e-6)
    assert 0 < ours["outlier_pct"] < 100  # the comparison is not vacuous


def test_outlier_reference_magnitude_differs():
    gt = np.zeros((2, H, W))
    pred = np.zeros((2, H, W))
    gt[0, 0, 0], pred[0, 0, 0] = 100.0, 95.2   # EPE 4.8: < 5 % of |GT|, > 5 % of |pred|
    valid = np.zeros((H, W), bool)
    valid[0, 0] = True
    x, y, t, p, _ = _events(0, n=10)
    _, frac_out = _oracle(ef.AEE, pred, _oracle_inputs(x, y, t, p, event_mask=valid, gt=gt))
    ours = metrics_ref.flow_metrics(pred, gt, valid_gt=valid)
    assert ours["outlier_pct"] == 0.0 and float(frac_out) == pytest.approx(1.0)


def test_src_never_imports_oracles():
    for path in (REPO / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any("oracles" in n for n in names), f"{path} imports {names}"
