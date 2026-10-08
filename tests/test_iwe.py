"""Image of warped events, FWL and RSAT on the analytic moving-edge fixture (§1.6, checklist C2)."""

import numpy as np
import pytest
import torch

from evflow.events.representations_ref import count_image
from evflow.flow import fwl as tf
from evflow.flow import iwe as ti
from evflow.flow import iwe_ref as ref

from fixtures.moving_edge import moving_edges

INTERP = ["round", "bilinear"]


@pytest.fixture(scope="module")
def edge():
    return moving_edges()


def _args(ev, **kw):
    return dict(t_range=(ev.t_start_us, ev.t_end_us), height=ev.height, width=ev.width, **kw)


def _t(*arrs):
    return [torch.from_numpy(np.asarray(a).astype(np.int64) if np.asarray(a).dtype == np.uint16
                             else np.asarray(a)) for a in arrs]


def test_warp_sign_and_units():
    # flow = +4 px over the window in x, -2 in y; an event at window start moves by the full flow
    # to t_ref = 1 (window end); an event at the end moves back by it to t_ref = 0.
    xw, yw = ref.warp(np.array([10.0, 10.0]), np.array([5.0, 5.0]), np.array([0.0, 1.0]),
                      np.array([[4.0, -2.0], [4.0, -2.0]]), t_ref=1.0)
    np.testing.assert_allclose(xw, [14.0, 10.0])
    np.testing.assert_allclose(yw, [3.0, 5.0])
    xw, _ = ref.warp(np.array([10.0]), np.array([5.0]), np.array([1.0]),
                     np.array([[4.0, -2.0]]), t_ref=0.0)
    np.testing.assert_allclose(xw, [6.0])


def test_normalize_time_window():
    tn = ref.normalize_time(np.array([100, 150, 199]), (100, 200))
    np.testing.assert_allclose(tn, [0.0, 0.5, 0.99])


@pytest.mark.parametrize("interp", INTERP)
def test_zero_flow_iwe_is_event_count(edge, interp):
    zero = np.zeros((2, edge.height, edge.width))
    img = ref.iwe(edge.x, edge.y, edge.t, zero, **_args(edge, t_ref=1.0, interpolation=interp))
    cnt = count_image(edge.x, edge.y, edge.p, height=edge.height, width=edge.width).sum(0)
    np.testing.assert_array_equal(img, cnt)


@pytest.mark.parametrize("interp", INTERP)
@pytest.mark.parametrize("t_ref", [0.0, 0.5, 1.0])
def test_fwl_zero_flow_is_exactly_one(edge, interp, t_ref):
    zero = np.zeros((2, edge.height, edge.width))
    a = _args(edge, t_ref=t_ref, interpolation=interp)
    assert ref.fwl(edge.x, edge.y, edge.t, zero, **a) == 1.0
    x, y, t = _t(edge.x, edge.y, edge.t)
    assert tf.fwl(x, y, t, torch.zeros(2, edge.height, edge.width, dtype=torch.float64), **a) == 1.0


@pytest.mark.parametrize("interp", INTERP)
def test_fwl_correct_flow_beats_zero_and_wrong_sign(edge, interp):
    a = _args(edge, t_ref=1.0, interpolation=interp)
    good = ref.fwl(edge.x, edge.y, edge.t, edge.flow_px, **a)
    wrong = ref.fwl(edge.x, edge.y, edge.t, -edge.flow_px, **a)
    assert good > 1.0
    assert good > wrong
    for s in (0.5, 1.5):
        assert good > ref.fwl(edge.x, edge.y, edge.t, s * edge.flow_px, **a)
    swapped = edge.flow_px[::-1].copy()  # u <-> v swap must also lose
    assert good > ref.fwl(edge.x, edge.y, edge.t, swapped, **a)


@pytest.mark.parametrize("interp", INTERP)
def test_rsat_zero_is_one_and_correct_below_one(edge, interp):
    a = _args(edge, t_ref=1.0, interpolation=interp)
    zero = np.zeros((2, edge.height, edge.width))
    assert ref.rsat(edge.x, edge.y, edge.t, edge.p, zero, **a) == 1.0
    good = ref.rsat(edge.x, edge.y, edge.t, edge.p, edge.flow_px, **a)
    wrong = ref.rsat(edge.x, edge.y, edge.t, edge.p, -edge.flow_px, **a)
    assert good < 1.0 and good < wrong


def test_bilinear_iwe_conserves_in_bounds_mass():
    rng = np.random.default_rng(0)
    n, h, w = 400, 12, 15
    x = rng.integers(3, w - 3, n).astype(np.uint16)
    y = rng.integers(3, h - 3, n).astype(np.uint16)
    t = np.sort(rng.integers(0, 1000, n)).astype(np.int64)
    flow = rng.uniform(-2, 2, (2, h, w))
    img = ref.iwe(x, y, t, flow, t_range=(0, 1000), t_ref=0.5, height=h, width=w,
                  interpolation="bilinear")
    np.testing.assert_allclose(img.sum(), n)


def test_events_leaving_image_are_dropped():
    img = ref.iwe(np.array([0], np.uint16), np.array([0], np.uint16), np.array([0], np.int64),
                  np.full((2, 3, 3), -5.0), t_range=(0, 10), t_ref=1.0, height=3, width=3,
                  interpolation="round")
    assert img.sum() == 0


@pytest.mark.parametrize("interp", INTERP)
@pytest.mark.parametrize("seed", [0, 1])
def test_torch_equals_numpy(interp, seed):
    rng = np.random.default_rng(seed)
    n, h, w = 3000, 20, 26
    x = rng.integers(0, w, n).astype(np.uint16)
    y = rng.integers(0, h, n).astype(np.uint16)
    t = np.sort(rng.integers(5000, 25_000, n)).astype(np.int64)
    p = rng.integers(0, 2, n).astype(np.int8)
    flow = rng.normal(0, 3, (2, h, w))
    a = dict(t_range=(5000, 25_000), t_ref=1.0, height=h, width=w, interpolation=interp)
    xt, yt, tt, pt = _t(x, y, t, p)
    ft = torch.from_numpy(flow)
    np.testing.assert_allclose(ti.iwe(xt, yt, tt, ft, **a).numpy(),
                               ref.iwe(x, y, t, flow, **a), atol=1e-6)
    assert tf.fwl(xt, yt, tt, ft, **a) == pytest.approx(ref.fwl(x, y, t, flow, **a), abs=1e-6)
    assert tf.rsat(xt, yt, tt, pt, ft, **a) == pytest.approx(ref.rsat(x, y, t, p, flow, **a),
                                                             abs=1e-6)


def test_interpolation_must_be_explicit(edge):
    with pytest.raises(TypeError):
        ref.fwl(edge.x, edge.y, edge.t, edge.flow_px, t_range=(0, 1), t_ref=1.0,
                height=edge.height, width=edge.width)
    with pytest.raises(ValueError):
        ref.fwl(edge.x, edge.y, edge.t, edge.flow_px, **_args(edge, t_ref=1.0, interpolation="x"))
