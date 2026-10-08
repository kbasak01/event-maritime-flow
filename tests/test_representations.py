"""Event representations: numpy reference vs torch, plus analytic properties (§1.5)."""

import numpy as np
import pytest
import torch

from evflow.events import representations as rep
from evflow.events import representations_ref as ref

from fixtures.moving_edge import moving_edges

H, W = 6, 7


def _ev(n=500, seed=0, float_xy=False):
    rng = np.random.default_rng(seed)
    if float_xy:
        x = rng.uniform(-0.8, W - 0.2, n).astype(np.float64)
        y = rng.uniform(-0.8, H - 0.2, n).astype(np.float64)
    else:
        x = rng.integers(0, W, n).astype(np.uint16)
        y = rng.integers(0, H, n).astype(np.uint16)
    t = np.sort(rng.integers(1000, 21_000, n)).astype(np.int64)
    p = rng.integers(0, 2, n).astype(np.int8)
    return x, y, t, p


def _torch(*arrs):
    out = []
    for a in arrs:
        a = np.asarray(a)
        if a.dtype == np.uint16:
            a = a.astype(np.int64)
        out.append(torch.from_numpy(a))
    return out


# --- count -------------------------------------------------------------------------------------


def test_count_mass_conserved_and_channels():
    x, y, t, p = _ev()
    c = ref.count_image(x, y, p, height=H, width=W)
    assert c.shape == (2, H, W)
    assert c[0].sum() == np.sum(p == 0)  # channel = polarity value: 0 OFF, 1 ON
    assert c[1].sum() == np.sum(p == 1)
    assert (c[1] - c[0]).sum() == np.sum(2 * p.astype(int) - 1)  # signed mass


def test_count_float_coords_bilinear_mass():
    x, y, t, p = _ev(float_xy=True)
    c = ref.count_image(x, y, p, height=H, width=W)
    # Mass is conserved for events whose 4 neighbours are all inside the image.
    inner = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    c_in = ref.count_image(x[inner], y[inner], p[inner], height=H, width=W)
    assert np.isclose(c_in.sum(), inner.sum())
    assert c.sum() <= len(x) + 1e-9


def test_count_single_event_bilinear_weights():
    c = ref.count_image(np.array([2.25]), np.array([3.5]), np.array([1], np.int8),
                        height=H, width=W)
    np.testing.assert_allclose(c[1, 3, 2], 0.75 * 0.5)
    np.testing.assert_allclose(c[1, 3, 3], 0.25 * 0.5)
    np.testing.assert_allclose(c[1, 4, 2], 0.75 * 0.5)
    np.testing.assert_allclose(c[1, 4, 3], 0.25 * 0.5)
    assert c[0].sum() == 0


# --- voxel -------------------------------------------------------------------------------------


def test_voxel_bin_edges_and_time_bilinear():
    # window [0, 1000) with B=5: bin centres at t_norm = 0..4 -> t = 0, 250, 500, 750, 1000.
    x = np.array([1, 1, 1], np.uint16)
    y = np.array([2, 2, 2], np.uint16)
    t = np.array([0, 250, 625], np.int64)
    p = np.array([1, 0, 1], np.int8)
    v = ref.voxel_grid(x, y, t, p, bins=5, height=H, width=W, t_range=(0, 1000), normalize=False)
    assert v.shape == (5, H, W)
    np.testing.assert_allclose(v[:, 2, 1], [1.0, -1.0, 0.5, 0.5, 0.0])
    assert np.count_nonzero(v) == 4


def test_voxel_signed_mass_conserved():
    x, y, t, p = _ev()
    v = ref.voxel_grid(x, y, t, p, bins=5, height=H, width=W, t_range=(1000, 21_000),
                       normalize=False)
    np.testing.assert_allclose(v.sum(), np.sum(2 * p.astype(int) - 1), atol=1e-9)


def test_voxel_normalize_nonzero_mean_std():
    x, y, t, p = _ev()
    v = ref.voxel_grid(x, y, t, p, bins=5, height=H, width=W, t_range=(1000, 21_000),
                       normalize=True)
    nz = v[v != 0]
    np.testing.assert_allclose(nz.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(nz.std(ddof=1), 1.0, atol=1e-9)


def test_voxel_rejects_bad_range():
    x, y, t, p = _ev(n=3)
    with pytest.raises(ValueError):
        ref.voxel_grid(x, y, t, p, bins=5, height=H, width=W, t_range=(5, 5), normalize=False)


# --- time surface ------------------------------------------------------------------------------


def test_time_surface_latest_event_decay():
    x = np.array([0, 0, 3], np.uint16)
    y = np.array([0, 0, 1], np.uint16)
    t = np.array([100, 700, 400], np.int64)
    p = np.array([1, 1, 0], np.int8)
    ts = ref.time_surface(x, y, t, p, height=H, width=W, t_range=(0, 1000), tau_us=300.0)
    np.testing.assert_allclose(ts[1, 0, 0], np.exp(-(1000 - 700) / 300.0))
    np.testing.assert_allclose(ts[0, 1, 3], np.exp(-(1000 - 400) / 300.0))
    assert np.count_nonzero(ts) == 2


# --- torch == numpy ----------------------------------------------------------------------------


@pytest.mark.parametrize("float_xy", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("bins", [1, 5, 15])
def test_voxel_torch_equals_numpy(float_xy, normalize, bins):
    x, y, t, p = _ev(float_xy=float_xy, seed=bins)
    a = ref.voxel_grid(x, y, t, p, bins=bins, height=H, width=W, t_range=(1000, 21_000),
                       normalize=normalize)
    b = rep.voxel_grid(*_torch(x, y, t, p), bins=bins, height=H, width=W,
                       t_range=(1000, 21_000), normalize=normalize, dtype=torch.float64)
    np.testing.assert_allclose(b.numpy(), a, rtol=0, atol=1e-6)


@pytest.mark.parametrize("float_xy", [False, True])
def test_count_torch_equals_numpy(float_xy):
    x, y, t, p = _ev(float_xy=float_xy)
    a = ref.count_image(x, y, p, height=H, width=W)
    b = rep.count_image(*_torch(x, y, p), height=H, width=W, dtype=torch.float64)
    np.testing.assert_allclose(b.numpy(), a, rtol=0, atol=1e-6)


@pytest.mark.parametrize("float_xy", [False, True])
def test_time_surface_torch_equals_numpy(float_xy):
    x, y, t, p = _ev(float_xy=float_xy)
    a = ref.time_surface(x, y, t, p, height=H, width=W, t_range=(1000, 21_000), tau_us=20_000 / 3)
    b = rep.time_surface(*_torch(x, y, t, p), height=H, width=W, t_range=(1000, 21_000),
                         tau_us=20_000 / 3, dtype=torch.float64)
    np.testing.assert_allclose(b.numpy(), a, rtol=0, atol=1e-6)


def test_float32_close_to_float64_on_fixture():
    ev = moving_edges()
    args = dict(bins=5, height=ev.height, width=ev.width, t_range=(ev.t_start_us, ev.t_end_us),
                normalize=True)
    a = ref.voxel_grid(ev.x, ev.y, ev.t, ev.p, **args)
    b = rep.voxel_grid(*_torch(ev.x, ev.y, ev.t, ev.p), **args, dtype=torch.float32)
    np.testing.assert_allclose(b.numpy(), a, rtol=0, atol=1e-4)


def test_empty_window():
    e = np.zeros(0, np.uint16)
    v = ref.voxel_grid(e, e, np.zeros(0, np.int64), np.zeros(0, np.int8), bins=5, height=H,
                       width=W, t_range=(0, 10), normalize=True)
    assert v.shape == (5, H, W) and not v.any()
