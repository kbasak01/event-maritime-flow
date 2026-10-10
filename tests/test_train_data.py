"""Phase 5 training data: windows, split guard, augmentation consistency, dt mix, inputs."""

import numpy as np
import pytest
import torch
from test_cm_baseline import moving_dots

from evflow.events.representations import count_image, time_surface, voxel_grid
from evflow.flow.iwe import iwe
from evflow.sim.splits import LeakageError, load_split
from evflow.train.data import (
    Augment,
    Clip,
    ClipData,
    EventArrays,
    GTArray,
    WindowDataset,
    build_windows,
    clip_window_starts,
    collate,
    guard_split,
    in_channels,
    make_input,
    mixed_dt_weights,
    thin,
)

H, W = 48, 64
FRAME_US = 20_000
FLOW = (6.0, -4.0)


def synthetic(n_frames=12, n_sources=1, seed=0, hood_row=None, augment=None, dts=(1, 2)):
    """Moving dots at a constant flow per frame interval; window (k, dt) has GT dt * FLOW."""
    aps = np.arange(n_frames + 1, dtype=np.int64) * FRAME_US + 1_000_000
    sources = []
    for s in range(n_sources):
        xs, ys, ts, ps = [], [], [], []
        for k in range(n_frames):
            x, y, t, p = moving_dots(height=H, width=W, flow_px=FLOW, seed=seed + 100 * s + k,
                                     duration_us=FRAME_US)
            xs.append(x.numpy())
            ys.append(y.numpy())
            ts.append(t.numpy() + aps[k])
            ps.append(p.numpy())
        sources.append(EventArrays(np.concatenate(xs).astype(np.uint16),
                                   np.concatenate(ys).astype(np.uint16), np.concatenate(ts),
                                   np.concatenate(ps).astype(np.int8), source=f"syn{s}"))
    clip = Clip("syn", "mvsec_train", 0, n_frames)
    gts = {}
    for dt in dts:
        flow = np.zeros((2, H, W), np.float32)
        flow[0], flow[1] = dt * FLOW[0], dt * FLOW[1]
        gts[dt] = GTArray({k: (flow, np.ones((H, W), bool)) for k in range(n_frames)})
    cd = [ClipData(clip, sources)]
    windows = build_windows(cd, aps, dts, 2, gts)
    ds = WindowDataset(cd, windows, gts, augment=augment, hood_row=hood_row, height=H, width=W)
    return ds, aps


def test_clip_window_starts_drop_warmup_and_stay_inside():
    c = Clip("c", "mvsec_train", 100, 200)
    k1 = clip_window_starts(c, 1, 10)
    k4 = clip_window_starts(c, 4, 10)
    assert k1[0] == 110 and k1[-1] == 199 and k4[-1] == 196
    with pytest.raises(ValueError):
        clip_window_starts(Clip("c", "mvsec_train", 0, 5), 4, 10)


def test_split_guard_refuses_test_and_out_of_range():
    train = load_split("mvsec_train")
    a = train["aps"][1]
    guard_split("mvsec_train", Clip("ok", "mvsec_train", a["aps_first"], a["aps_first"] + 50))
    with pytest.raises(LeakageError):
        guard_split("mvsec_test", Clip("x", "mvsec_test", 10, 20))
    tune = load_split("mvsec_tune")["aps"][0]
    with pytest.raises(LeakageError):   # tune frames are not training frames
        guard_split("mvsec_train", Clip("x", "mvsec_train", tune["aps_first"],
                                        tune["aps_first"] + 10))


def test_windows_and_items():
    ds, aps = synthetic()
    assert ds.dt_sizes == {1: 10, 2: 9}
    it = ds[0]
    assert it["dt"] == 1 and it["k"] == 2 and it["duration_us"] == FRAME_US
    assert it["t"].min() >= 0 and it["t"].max() < FRAME_US
    assert torch.allclose(it["flow"][0], torch.full((H, W), FLOW[0]))
    it2 = ds[ds.dt_sizes[1]]
    assert it2["dt"] == 2 and it2["duration_us"] == 2 * FRAME_US
    assert torch.allclose(it2["flow"][1], torch.full((H, W), 2 * FLOW[1]))


def test_hood_rows_masked_and_excluded_from_cm():
    ds, _ = synthetic(hood_row=40)
    it = ds[0]
    assert not it["mask"][40:].any() and it["mask"][:40].all()
    assert torch.equal(it["cm_keep"], it["y"] < 40)


def test_crop_shifts_events_and_flow():
    ds, _ = synthetic(hood_row=40, augment=Augment(crop=(32, 48), hflip=False,
                                                    fixed_crop=(8, 10)))
    full, _ = synthetic(hood_row=40)
    it, ref = ds[0], full[0]
    keep = (ref["x"] >= 10) & (ref["x"] < 58) & (ref["y"] >= 8) & (ref["y"] < 40)
    assert torch.equal(it["x"], ref["x"][keep] - 10) and torch.equal(it["y"], ref["y"][keep] - 8)
    assert it["flow"].shape == (2, 32, 48) and it["mask"].shape == (32, 48)
    assert not it["mask"][32:].any() and it["mask"][:32].all()   # hood at full row 40 = crop 32


def test_hflip_consistent_with_warping():
    """IWE of flipped events under the flipped flow = the flipped IWE of the original."""
    ref_ds, _ = synthetic()
    ds, _ = synthetic(augment=Augment(crop=None, hflip=True))
    ref = ref_ds[0]
    flipped = None
    for s in range(20):
        torch.manual_seed(s)
        it = ds[0]
        if not torch.equal(it["x"], ref["x"]):
            flipped = it
            break
    assert flipped is not None
    assert torch.equal(flipped["x"], W - 1 - ref["x"])
    assert torch.allclose(flipped["flow"][0], -ref["flow"][0])
    a = dict(t_range=(0, FRAME_US), t_ref=1.0, height=H, width=W, interpolation="bilinear")
    img = iwe(ref["x"], ref["y"], ref["t"], ref["flow"].double(), **a)
    img_f = iwe(flipped["x"], flipped["y"], flipped["t"], flipped["flow"].double(), **a)
    assert torch.allclose(img_f, torch.flip(img, dims=[1]), atol=1e-9)


def test_random_ct_source_drawn():
    ds, _ = synthetic(n_sources=4)
    torch.manual_seed(0)
    assert len({ds[0]["source"] for _ in range(40)}) == 4


def test_mixed_dt_weights():
    w = mixed_dt_weights({1: 100, 4: 25}, {1: 0.5, 4: 0.5})
    assert float(w[:100].sum()) == pytest.approx(0.5) and float(w[100:].sum()) == pytest.approx(0.5)
    g = torch.Generator().manual_seed(0)
    draws = torch.multinomial(w, 20_000, replacement=True, generator=g)
    assert float((draws >= 100).double().mean()) == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        mixed_dt_weights({1: 10, 4: 10}, {1: 0.7, 4: 0.5})


def test_thin_evenly_spaced():
    ds, _ = synthetic()
    w = {1: list(range(10))}
    assert thin(w, 3)[1] == [0, 4, 9] or thin(w, 3)[1] == [0, 5, 9]
    assert thin(w, None) is w
    del ds


def test_make_input_dispatch():
    ds, _ = synthetic()
    it = ds[0]
    x, y, t, p = it["x"], it["y"], it["t"], it["p"]
    tr = (0, FRAME_US)
    assert torch.equal(make_input("voxel5", x, y, t, p, duration_us=FRAME_US, height=H, width=W),
                       voxel_grid(x, y, t, p, bins=5, height=H, width=W, t_range=tr,
                                  normalize=True))
    assert torch.equal(make_input("count", x, y, t, p, duration_us=FRAME_US, height=H, width=W),
                       count_image(x, y, p, height=H, width=W))
    assert torch.equal(
        make_input("timesurface", x, y, t, p, duration_us=FRAME_US, height=H, width=W),
        time_surface(x, y, t, p, height=H, width=W, t_range=tr, tau_us=FRAME_US / 3))
    assert [in_channels(r) for r in ("voxel5", "voxel15", "count", "timesurface")] == [5, 15, 2, 2]
    with pytest.raises(ValueError):
        in_channels("voxel_fancy")


def test_collate_keeps_variable_length_events():
    ds, _ = synthetic()
    b = collate([ds[0], ds[1], ds[12]])
    assert b["flow"].shape == (3, 2, H, W) and b["n_events"].numel() == 3
    n0 = ds[0]["x"].numel()
    assert torch.equal(b["ev_x"][:n0], ds[0]["x"]) and int((b["ev_b"] == 0).sum()) == n0
    assert b["dt"].tolist() == [1, 1, 2]


@pytest.mark.parametrize("rep", ["voxel5", "voxel15", "count", "timesurface"])
def test_batch_inputs_equal_per_sample(rep):
    from evflow.train.data import batch_inputs
    ds, _ = synthetic()
    items = [ds[0], ds[3], ds[12]]
    items[1] = {**items[1], **{k: items[1][k][:0] for k in ("x", "y", "t", "p")}}  # empty window
    b = collate(items)
    got = batch_inputs(b, rep, H, W, "cpu")
    ref = torch.stack([make_input(rep, it["x"], it["y"], it["t"], it["p"],
                                  duration_us=it["duration_us"], height=H, width=W)
                       for it in items])
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
