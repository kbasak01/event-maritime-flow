"""M0b: per-window patch-wise contrast maximisation on analytic moving-edge fixtures."""

import numpy as np
import pytest
import torch

from evflow.flow.fwl import fwl
from evflow.models.cm_baseline import CMParams, estimate_flow, tiles_to_flow
from evflow.models.zero_flow import zero_flow

from fixtures.moving_edge import moving_edges

FAST = CMParams(tile_px=16, iters_global=90, iters_tiles=60, lr=0.4, tv_weight=0.01,
                sigmas=(3.0, 1.5, 1.0), max_events=200_000)


def moving_dots(*, height, width, flow_px, n_dots=200, events_per_dot=30, seed=0,
                duration_us=20_000):
    """Sparse random dots translating by ``flow_px`` over the window; each dot fires
    ``events_per_dot`` events at uniform random times, at its position rounded to a pixel. Unlike
    straight edges, dots have no aperture problem, so the flow is fully observable."""
    rng = np.random.default_rng(seed)
    w = np.asarray(flow_px, np.float64)
    p0 = rng.uniform([-abs(w[0]), -abs(w[1])], [width + abs(w[0]), height + abs(w[1])],
                     (n_dots, 2))
    tau = rng.uniform(0.0, 1.0, (n_dots, events_per_dot))
    pos = (p0[:, None, :] + tau[..., None] * w).reshape(-1, 2)
    xi, yi = np.round(pos[:, 0]), np.round(pos[:, 1])
    ok = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    t = (tau.reshape(-1)[ok] * duration_us).astype(np.int64)
    o = np.argsort(t, kind="stable")
    x, y, t = (torch.from_numpy(a[o].astype(np.int64)) for a in (xi[ok], yi[ok], t))
    return x, y, t, torch.ones_like(x)


def _tensors(e, x_offset=0):
    return (torch.from_numpy(e.x.astype(np.int64) + x_offset),
            torch.from_numpy(e.y.astype(np.int64)), torch.from_numpy(e.t),
            torch.from_numpy(e.p.astype(np.int64)))


def _run(x, y, t, p, *, h, w, t_range, params=FAST):
    torch.manual_seed(0)
    return estimate_flow(x, y, t, p, t_range=t_range, height=h, width=w, params=params,
                         device="cpu")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_translation_recovered_dots(seed):
    x, y, t, p = moving_dots(height=48, width=64, flow_px=(8.0, -5.0), seed=seed)
    flow = _run(x, y, t, p, h=48, w=64, t_range=(0, 20_000))
    assert flow.shape == (2, 48, 64)
    inner = flow[:, 8:-8, 8:-8].reshape(2, -1).median(dim=1).values.double()
    assert torch.allclose(inner, torch.tensor([8.0, -5.0], dtype=torch.float64), atol=0.2)


def test_translation_moving_edges():
    """Straight edges at pixel-centre firing times: the contrast optimum of this discrete fixture
    itself sits ~0.4 px from the analytic flow (grid search, Phase 4), hence the looser bound."""
    e = moving_edges(width=64, height=48, velocity_px_per_s=(400.0, -250.0), n_edges=10)
    x, y, t, p = _tensors(e)
    flow = _run(x, y, t, p, h=48, w=64, t_range=(e.t_start_us, e.t_end_us))
    inner = flow[:, 8:-8, 8:-8].reshape(2, -1).median(dim=1).values.double()
    assert torch.allclose(inner, torch.tensor([8.0, -5.0], dtype=torch.float64), atol=0.5)


def test_two_regions_recovered():
    """Left half moves right/up, right half moves left/down: one global vector cannot fit both."""
    xa, ya, ta, pa = moving_dots(height=48, width=48, flow_px=(6.0, -4.0), seed=1)
    xb, yb, tb, pb = moving_dots(height=48, width=48, flow_px=(-5.0, 3.0), seed=2)
    t = torch.cat([ta, tb])
    order = torch.argsort(t, stable=True)
    x = torch.cat([xa, xb + 48])[order]
    y, p = torch.cat([ya, yb])[order], torch.cat([pa, pb])[order]
    flow = _run(x, y, t[order], p, h=48, w=96, t_range=(0, 20_000),
                params=CMParams.from_dict({**FAST.to_dict(), "iters_tiles": 120}))
    left = flow[:, 12:36, 8:28].reshape(2, -1).median(dim=1).values.double()
    right = flow[:, 12:36, 68:88].reshape(2, -1).median(dim=1).values.double()
    assert torch.allclose(left, torch.tensor([6.0, -4.0], dtype=torch.float64), atol=0.5)
    assert torch.allclose(right, torch.tensor([-5.0, 3.0], dtype=torch.float64), atol=0.5)


def test_empty_window_gives_zero_flow():
    e = torch.zeros(0, dtype=torch.int64)
    flow = _run(e, e, e, e, h=20, w=30, t_range=(0, 1000))
    assert flow.shape == (2, 20, 30)
    assert torch.count_nonzero(flow) == 0


def test_deterministic_on_cpu():
    e = moving_edges(width=40, height=32, n_edges=6, seed=3)
    args = (*_tensors(e),)
    kw = dict(h=32, w=40, t_range=(e.t_start_us, e.t_end_us))
    assert torch.equal(_run(*args, **kw), _run(*args, **kw))


def test_estimated_flow_sharpens_iwe():
    e = moving_edges(width=64, height=48, n_edges=10, seed=4)
    x, y, t, p = _tensors(e)
    tr = (e.t_start_us, e.t_end_us)
    flow = _run(x, y, t, p, h=48, w=64, t_range=tr).double()
    a = dict(t_range=tr, t_ref=1.0, height=48, width=64, interpolation="round")
    assert fwl(x, y, t, flow, **a) > 1.2
    assert fwl(x, y, t, zero_flow(48, 64), **a) == pytest.approx(1.0, abs=0.0)


def test_tiles_to_flow_constant_and_shape():
    tiles = torch.ones(2, 3, 4)
    tiles[1] *= -2.0
    flow = tiles_to_flow(tiles, height=40, width=50, tile_px=16)
    assert flow.shape == (2, 40, 50)
    assert torch.allclose(flow[0], torch.ones(40, 50))
    assert torch.allclose(flow[1], torch.full((40, 50), -2.0))


def test_params_reject_bad_values():
    with pytest.raises(ValueError):
        CMParams(tile_px=0)
    with pytest.raises(ValueError):
        CMParams(t_refs=())


@pytest.mark.parametrize("floating", [False, True])
def test_iwe_stack_equals_flow_iwe(floating):
    from evflow.flow.iwe import normalize_time, sample_flow, splat, warp
    from evflow.models.cm_baseline import iwe_stack

    rng = np.random.default_rng(5)
    h, w, pad, n = 20, 30, 7, 3000
    x = torch.from_numpy(rng.integers(0, w, n))
    y = torch.from_numpy(rng.integers(0, h, n))
    if floating:
        x = x.double() + torch.from_numpy(rng.uniform(-0.6, 0.4, n))
        y = y.double() + torch.from_numpy(rng.uniform(-0.6, 0.4, n))
    t = torch.from_numpy(np.sort(rng.integers(0, 10_000, n)))
    flow = torch.from_numpy(rng.normal(0, 4, (2, h, w)))
    tau = normalize_time(t, (0, 10_000))
    got = iwe_stack(x, y, tau, flow, t_refs=(0.0, 0.5, 1.0), height=h, width=w, pad=pad)
    fe = sample_flow(flow, x, y)  # flow.iwe semantics: own pixel, clamped to the sensor
    for k, tr in enumerate((0.0, 0.5, 1.0)):
        xw, yw = warp(x, y, tau, fe, t_ref=tr)
        ref = splat(xw + pad, yw + pad, height=h + 2 * pad, width=w + 2 * pad,
                    interpolation="bilinear")
        assert torch.allclose(got[k], ref, atol=1e-9)
