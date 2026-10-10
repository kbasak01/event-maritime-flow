"""Contrast-maximisation loss and its event-collapse guards, on analytic fixtures."""

import numpy as np
import pytest
import torch
from test_cm_baseline import moving_dots

from evflow.losses.cm import (
    CMLossParams,
    WindowEvents,
    cm_diagnostics,
    cm_loss,
    divergence,
    subsample,
)

from fixtures.moving_edge import moving_edges

H, W = 48, 64
FLOW = (6.0, -4.0)
GUARDED = CMLossParams(divergence_penalty=1.0)
UNGUARDED = CMLossParams(reference_times=(1.0,), smoothness_weight=0.0, divergence_penalty=0.0)


def _dots(seed):
    x, y, t, _ = moving_dots(height=H, width=W, flow_px=FLOW, seed=seed)
    return WindowEvents(x, y, (t.double() / 20_000).float())


def _const(u, v):
    f = torch.zeros(1, 2, H, W)
    f[0, 0], f[0, 1] = u, v
    return f


def _contracting(a):
    """True translation plus a field contracting (a > 0) or expanding (a < 0) about the centre:
    it can pile the events of one reference time onto few pixels (event collapse)."""
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    f = _const(*FLOW)
    f[0, 0] -= a * (xx - W / 2)
    f[0, 1] -= a * (yy - H / 2)
    return f


def _loss(flow, ev, p):
    return float(cm_loss(flow, [ev], p)[0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_true_flow_beats_zero_beats_reverse(seed):
    ev = _dots(seed)
    assert _loss(_const(*FLOW), ev, GUARDED) < _loss(_const(0, 0), ev, GUARDED) \
        < _loss(_const(-FLOW[0], -FLOW[1]), ev, GUARDED)


def test_true_flow_beats_zero_on_moving_edges():
    e = moving_edges(width=W, height=H, velocity_px_per_s=(300.0, -200.0), n_edges=10)
    tau = torch.from_numpy((e.t - e.t_start_us) / (e.t_end_us - e.t_start_us)).float()
    ev = WindowEvents(torch.from_numpy(e.x.astype(np.int64)),
                      torch.from_numpy(e.y.astype(np.int64)), tau)
    true = torch.from_numpy(e.flow_px).float()[None]
    assert _loss(true, ev, GUARDED) < _loss(torch.zeros_like(true), ev, GUARDED)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unguarded_objective_prefers_collapse(seed):
    """Single reference time, no regulariser: a contracting field beats the true flow (the
    failure mode the guards exist for)."""
    ev = _dots(seed)
    assert _loss(_contracting(1.0), ev, UNGUARDED) < _loss(_const(*FLOW), ev, UNGUARDED) - 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_guarded_objective_rejects_collapse(seed):
    ev = _dots(seed)
    true = _loss(_const(*FLOW), ev, GUARDED)
    for a in np.linspace(-3, 3, 49):
        if abs(a) > 1e-9:
            assert _loss(_contracting(float(a)), ev, GUARDED) > true, a


def test_divergence_of_linear_field():
    f = _contracting(0.5)[0]
    assert torch.allclose(divergence(f), torch.full((H - 1, W - 1), -1.0), atol=1e-5)
    assert float(divergence(_const(3, 2)[0]).abs().max()) == 0.0


def test_gradient_points_to_true_flow():
    ev = _dots(0)
    flow = _const(0.5 * FLOW[0], 0.5 * FLOW[1]).requires_grad_(True)
    cm_loss(flow, [ev], GUARDED)[0].backward()
    g = flow.grad[0].mean(dim=(1, 2))
    assert g[0] < 0 and g[1] > 0          # descent increases u towards 6, decreases v towards -4


def test_empty_sample_has_no_data_term():
    empty = WindowEvents(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                         torch.zeros(0))
    loss, parts = cm_loss(_const(1, 1), [empty], GUARDED)
    assert parts["cm_data"] == 0.0
    assert float(loss) == pytest.approx(GUARDED.smoothness_weight * parts["cm_smooth"])


def test_subsample_bounded_and_deterministic():
    ev = _dots(0)
    a, b = subsample(ev, 100), subsample(ev, 100)
    assert a.tau.numel() == 100 and torch.equal(a.x, b.x)
    assert subsample(ev, 10**9) is ev


def test_diagnostics_flag_collapse():
    ev = _dots(0)
    good = cm_diagnostics(_const(*FLOW)[0], ev)
    bad = cm_diagnostics(_contracting(1.0)[0], ev)
    assert good["abs_div"] == 0.0 and bad["abs_div"] == pytest.approx(2.0, rel=1e-4)
    assert bad["flow_mag_events"] > 2 * good["flow_mag_events"]
    zero = cm_diagnostics(_const(0, 0)[0], ev)
    assert zero["iwe_nonempty_ratio"] == pytest.approx(1.0)
    assert zero["iwe_var_ratio"] == pytest.approx(1.0)
