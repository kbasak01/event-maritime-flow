"""Self-supervised motion-compensation losses and their event-collapse guards, on analytic
fixtures: the average-timestamp objective (D-047) and the superseded IWE-variance one (D-043)."""

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
GUARDED = CMLossParams(objective="iwe_variance", divergence_penalty=1.0)
UNGUARDED = CMLossParams(objective="iwe_variance", reference_times=(1.0,), smoothness_weight=0.0,
                         divergence_penalty=0.0)
AT = CMLossParams(objective="average_timestamp", smoothness_weight=0.1, divergence_penalty=0.3)


def _dots(seed):
    x, y, t, _ = moving_dots(height=H, width=W, flow_px=FLOW, seed=seed)
    p = torch.from_numpy(np.random.default_rng(seed).integers(0, 2, x.numel()))
    return WindowEvents(x, y, (t.double() / 20_000).float(), p)


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
    assert a.p is not None and a.p.numel() == 100


def test_diagnostics_flag_collapse():
    ev = _dots(0)
    good = cm_diagnostics(_const(*FLOW)[0], ev)
    bad = cm_diagnostics(_contracting(1.0)[0], ev)
    assert good["abs_div"] == 0.0 and bad["abs_div"] == pytest.approx(2.0, rel=1e-4)
    assert bad["flow_mag_events"] > 2 * good["flow_mag_events"]
    zero = cm_diagnostics(_const(0, 0)[0], ev)
    assert zero["iwe_nonempty_ratio"] == pytest.approx(1.0)
    assert zero["iwe_var_ratio"] == pytest.approx(1.0)


# --- average timestamp (D-047) -------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_at_true_flow_beats_zero_beats_reverse(seed):
    ev = _dots(seed)
    assert _loss(_const(*FLOW), ev, AT) < _loss(_const(0, 0), ev, AT) \
        < _loss(_const(-FLOW[0], -FLOW[1]), ev, AT)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_at_guarded_rejects_collapse(seed):
    ev = _dots(seed)
    true = _loss(_const(*FLOW), ev, AT)
    for a in np.linspace(-3, 3, 49):
        if abs(a) > 1e-9:
            assert _loss(_contracting(float(a)), ev, AT) > true, a


def test_at_images_conserve_events_and_weights():
    from evflow.losses.cm import average_timestamp_images
    ev = _dots(0)
    cnt, ts = average_timestamp_images(ev, _const(*FLOW)[0], t_ref=1.0, weights=ev.tau, pad=64)
    assert float(cnt.sum()) == pytest.approx(ev.tau.numel(), rel=1e-5)   # padded canvas keeps all
    assert float(ts.sum()) == pytest.approx(float(ev.tau.sum()), rel=1e-5)
    for pol in (0, 1):
        assert float(cnt[pol].sum()) == pytest.approx(float((ev.p == pol).sum()), rel=1e-5)


def test_at_needs_polarity_and_objective_validated():
    from evflow.losses.cm import average_timestamp_loss
    ev = _dots(0)
    with pytest.raises(ValueError):
        average_timestamp_loss(WindowEvents(ev.x, ev.y, ev.tau), _const(*FLOW)[0], 64)
    with pytest.raises(ValueError):
        CMLossParams(objective="contrast")


def test_at_minimum_at_true_flow_on_grid():
    """Global minimum of the data term over a 1 px grid of constant flows is the true flow.
    (The basin is narrow on this sparse fixture: from half the true flow the gradient need not
    point at it, so no gradient-direction test is made for this objective.)"""
    from evflow.losses.cm import average_timestamp_loss
    ev = _dots(0)
    grid = [(u, v) for u in range(0, 10) for v in range(-7, 2)]
    vals = [float(average_timestamp_loss(ev, _const(u, v)[0], 64)) for u, v in grid]
    assert grid[int(np.argmin(vals))] == (6, -4)


# --- low-count floor (D-049 follow-up, D-056) ------------------------------------------------

AT_FLOOR = CMLossParams(objective="average_timestamp", smoothness_weight=0.1,
                        divergence_penalty=0.3, min_count=1e-2)


def test_at_floor_zero_is_the_phase5_loss():
    from evflow.losses.cm import average_timestamp_images, average_timestamp_loss
    ev = _dots(1)
    f = _const(2.3, -1.7)[0]
    ref = 0.0
    for t_ref, wts in ((1.0, ev.tau), (0.0, 1.0 - ev.tau)):
        cnt, ts = average_timestamp_images(ev, f, t_ref=t_ref, weights=wts, pad=64)
        avg = ts / (cnt + 1e-9)
        ref += float((avg * avg).sum() / (cnt.sum(0) > 0).sum())
    assert float(average_timestamp_loss(ev, f, 64)) == pytest.approx(ref, rel=1e-6)
    # bit-identical to the Phase 5 code path (min_count default 0)
    assert torch.equal(average_timestamp_loss(ev, f, 64, 0.0), average_timestamp_loss(ev, f, 64))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_at_floor_keeps_the_minimum_and_rejects_collapse(seed):
    ev = _dots(seed)
    assert _loss(_const(*FLOW), ev, AT_FLOOR) < _loss(_const(0, 0), ev, AT_FLOOR) \
        < _loss(_const(-FLOW[0], -FLOW[1]), ev, AT_FLOOR)
    true = _loss(_const(*FLOW), ev, AT_FLOOR)
    for a in np.linspace(-3, 3, 25):
        if abs(a) > 1e-9:
            assert _loss(_contracting(float(a)), ev, AT_FLOOR) > true, a


def test_at_floor_bounds_the_low_count_gradient():
    """An event whose bilinear taps put ~1e-4 weight on a pixel no other event reaches: without
    the floor that pixel's ratio has a gradient ~1/count; with it the cell is ignored."""
    from evflow.losses.cm import average_timestamp_loss
    x = torch.tensor([10, 30]).long()
    y = torch.tensor([10, 20]).long()
    ev = WindowEvents(x, y, torch.tensor([0.2, 0.7]), torch.tensor([1, 0]))
    f = torch.zeros(2, H, W)
    f[0] = 1e-4 / 0.8          # event 0 moves 1e-4 px by t_ref = 1: a 1e-4 tap on x = 11
    f.requires_grad_(True)
    g0 = torch.autograd.grad(average_timestamp_loss(ev, f, 64), f)[0].abs().max()
    f2 = f.detach().clone().requires_grad_(True)
    g1 = torch.autograd.grad(average_timestamp_loss(ev, f2, 64, 1e-2), f2)[0].abs().max()
    assert float(g1) < 1e-2 * float(g0)
