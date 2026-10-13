"""MVSEC converter and GT-flow reader on fake files with the observed layout (D-015, D-019)."""

import h5py
import numpy as np
import pytest

from evflow.io import mvsec
from evflow.io.store import EventSlicer, EventStore, compute_ms_to_idx

from fixtures.fake_sources import write_fake_mvsec


@pytest.fixture
def fake(tmp_path):
    src = tmp_path / "seq_data.hdf5"
    info = write_fake_mvsec(src)
    return src, info


def test_convert_events_mapping(fake, tmp_path):
    src, info = fake
    dst = tmp_path / "out.h5"
    summary = mvsec.convert_events(src, dst, sequence="fake_seq", chunk=777)
    ev = info["events"]
    t_abs = np.round(ev[:, 2] * 1e6).astype(np.int64)
    assert summary["n_events"] == len(ev)
    assert summary["t_offset_us"] == t_abs[0]
    with EventStore(dst) as s:
        assert (s.width, s.height) == (346, 260)
        assert s.attrs["distortion"] == "raw" and s.attrs["sensor"] == "DAVIS346B"
        assert s.attrs["source"] == "mvsec/fake_seq/davis/left"
        t = s.events["t"][:]
        np.testing.assert_array_equal(t, t_abs - t_abs[0])
        np.testing.assert_array_equal(s.events["p"][:], (ev[:, 3] > 0).astype(np.int8))
        np.testing.assert_array_equal(s.events["x"][:], ev[:, 0].astype(np.uint16))
        np.testing.assert_array_equal(s.events["y"][:], ev[:, 1].astype(np.uint16))
        np.testing.assert_array_equal(s.ms_to_idx, compute_ms_to_idx(t))
    # absolute-time slicing works on the MVSEC clock (µs since epoch)
    with EventSlicer(dst) as sl:
        a, b = t_abs[100], t_abs[900]
        got = sl.get_events(int(a), int(b))
        assert len(got["t"]) == np.sum((t_abs >= a) & (t_abs < b))


def test_convert_rejects_non_integer_xy(tmp_path):
    src = tmp_path / "bad.hdf5"
    write_fake_mvsec(src)
    with h5py.File(src, "r+") as f:
        f["davis/left/events"][3, 0] = 10.5
    with pytest.raises(ValueError, match="integer"):
        mvsec.convert_events(src, tmp_path / "o.h5", sequence="bad")


def test_convert_rejects_bad_polarity(tmp_path):
    src = tmp_path / "bad.hdf5"
    write_fake_mvsec(src)
    with h5py.File(src, "r+") as f:
        f["davis/left/events"][3, 3] = 0.0
    with pytest.raises(ValueError, match="polarity"):
        mvsec.convert_events(src, tmp_path / "o.h5", sequence="bad")


def test_convert_rejects_decreasing_time(tmp_path):
    src = tmp_path / "bad.hdf5"
    write_fake_mvsec(src)
    with h5py.File(src, "r+") as f:
        f["davis/left/events"][2000, 2] = f["davis/left/events"][0, 2]
    with pytest.raises(ValueError, match="non-decreasing"):
        mvsec.convert_events(src, tmp_path / "o.h5", sequence="bad", chunk=500)


def test_aps_timestamps_us(fake):
    src, info = fake
    ts = mvsec.read_aps_timestamps_us(src)
    assert ts.dtype == np.int64
    np.testing.assert_array_equal(ts, np.round(info["aps_ts"] * 1e6).astype(np.int64))


# --- GT flow -----------------------------------------------------------------------------------

H, W = 30, 40
GT_DT_US = 50_000
T0_US = 1_504_645_177_711_321


def _gt_npz(path, fields):
    ts = (T0_US + np.arange(len(fields)) * GT_DT_US) * 1e-6
    np.savez(path, timestamps=ts, x_flow_dist=np.stack([f[0] for f in fields]),
             y_flow_dist=np.stack([f[1] for f in fields]))
    return mvsec.GTFlow(path)


def _uniform(u, v, n=8):
    f = np.zeros((2, H, W))
    f[0], f[1] = u, v
    return [f.copy() for _ in range(n)]


def test_gt_reader_memmaps_stored_npz(tmp_path):
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(2, H, W)) for _ in range(5)]
    gt = _gt_npz(tmp_path / "g.npz", fields)
    assert gt.memmapped and len(gt) == 5
    np.testing.assert_array_equal(gt.field(3), fields[3])
    assert gt.timestamps_us[1] - gt.timestamps_us[0] == GT_DT_US


def test_gt_short_window_is_scaled(tmp_path):
    gt = _gt_npz(tmp_path / "g.npz", _uniform(2.0, -1.0))
    flow, valid = mvsec.gt_flow_for_window(gt, T0_US + 10_000, T0_US + 30_000)  # 20 of 50 ms
    np.testing.assert_allclose(flow[:, 5, 5], [0.8, -0.4])
    assert valid.all()


def test_gt_four_intervals_uniform(tmp_path):
    gt = _gt_npz(tmp_path / "g.npz", _uniform(2.0, -1.0))
    flow, valid = mvsec.gt_flow_for_window(gt, T0_US, T0_US + 4 * GT_DT_US)
    np.testing.assert_allclose(flow[:, 15, 10], [8.0, -4.0])
    assert valid[15, 10]
    # pixels that leave the image (x + 8 > W-1 or y - 4 < 0 on the way) lose GT
    assert not valid[15, W - 1] and not valid[0, 10]


def test_gt_partial_intervals(tmp_path):
    gt = _gt_npz(tmp_path / "g.npz", _uniform(2.0, -1.0))
    start = T0_US + GT_DT_US // 4            # 75 % of interval 0
    end = T0_US + 3 * GT_DT_US + GT_DT_US // 2  # intervals 1, 2 full, 50 % of interval 3
    flow, valid = mvsec.gt_flow_for_window(gt, start, end)
    np.testing.assert_allclose(flow[:, 15, 10], np.array([2.0, -1.0]) * (0.75 + 2 + 0.5))


def test_gt_propagation_is_not_naive_scaling(tmp_path):
    f = np.zeros((2, H, W))
    f[0] = 0.1 * np.arange(W)[None, :]   # u grows with x; v = 0
    gt = _gt_npz(tmp_path / "g.npz", [f.copy() for _ in range(6)])
    flow, valid = mvsec.gt_flow_for_window(gt, T0_US, T0_US + 3 * GT_DT_US)
    x = 10.0
    for _ in range(3):  # nearest-pixel lookup of u at the propagated position
        x += 0.1 * np.round(x)
    assert flow[0, 7, 10] == pytest.approx(x - 10.0)
    assert flow[0, 7, 10] != pytest.approx(3 * 1.0)
    assert flow[1, 7, 10] == 0.0 and valid[7, 10]  # v == 0 zeroes only that component
    assert not valid[7, 0]                         # u == 0 at x = 0 and v == 0 -> no GT


def test_gt_window_outside_span_raises(tmp_path):
    gt = _gt_npz(tmp_path / "g.npz", _uniform(1.0, 1.0, n=3))
    with pytest.raises(ValueError):
        mvsec.gt_flow_for_window(gt, T0_US - 1, T0_US + 10)
    with pytest.raises(ValueError):
        mvsec.gt_flow_for_window(gt, T0_US, T0_US + 3 * GT_DT_US)


# --- Phase 10 restructuring (D-068): interval schedule + one advection loop -------------------

def test_interval_schedule_partial_and_full():
    ts = T0_US + GT_DT_US * np.arange(6, dtype=np.int64)
    start = T0_US + GT_DT_US // 4
    end = T0_US + 3 * GT_DT_US + GT_DT_US // 2
    assert mvsec._interval_schedule(ts, start, end) == [(0, 0.75), (1, 1.0), (2, 1.0), (3, 0.5)]


def test_interval_schedule_keeps_zero_length_last_interval():
    """There is always a first and a last step: a window of exactly one GT interval still looks
    up the next field with fraction 0, which only updates the masks (kept bit-identical)."""
    ts = T0_US + GT_DT_US * np.arange(6, dtype=np.int64)
    assert mvsec._interval_schedule(ts, T0_US, T0_US + GT_DT_US) == [(0, 1.0), (1, 0.0)]
    assert mvsec._interval_schedule(ts, T0_US, T0_US + 2 * GT_DT_US) == [(0, 1.0), (1, 1.0)]


def _reference_step_by_step(gt, t_start_us, t_end_us):
    """The pre-D-068 formulation (first / middle / last call sites), kept as the oracle."""
    import cv2

    def step(field, xs, ys, mx, my, scale):
        fx = cv2.remap(field[0], xs, ys, cv2.INTER_NEAREST)
        fy = cv2.remap(field[1], xs, ys, cv2.INTER_NEAREST)
        mx[fx == 0] = False
        my[fy == 0] = False
        xs += (fx * scale).astype(np.float32)
        ys += (fy * scale).astype(np.float32)

    ts = gt.timestamps_us
    it = int(np.searchsorted(ts, t_start_us, side="right")) - 1
    gt_dt = float(ts[it + 1] - ts[it])
    dt = float(t_end_us - t_start_us)
    if gt_dt > dt:
        flow = gt.field(it) * (dt / gt_dt)
    else:
        h, w = gt.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        x0, y0 = xs.copy(), ys.copy()
        mx, my = np.ones((h, w), bool), np.ones((h, w), bool)
        step(gt.field(it), xs, ys, mx, my, (ts[it + 1] - t_start_us) / gt_dt)
        it += 1
        while ts[it + 1] < t_end_us:
            step(gt.field(it), xs, ys, mx, my, 1.0)
            it += 1
        step(gt.field(it), xs, ys, mx, my, (t_end_us - ts[it]) / float(ts[it + 1] - ts[it]))
        flow = np.stack([np.where(mx, xs - x0, 0.0), np.where(my, ys - y0, 0.0)])
        flow = flow.astype(np.float64)
    valid = np.all(np.isfinite(flow), axis=0) & (np.hypot(flow[0], flow[1]) > 0)
    return flow, valid


def test_gt_flow_bit_identical_to_step_by_step_reference(tmp_path):
    rng = np.random.default_rng(7)
    fields = []
    for _ in range(8):
        f = rng.normal(scale=1.5, size=(2, H, W))
        f[rng.random((2, H, W)) < 0.1] = 0.0          # holes, as in MVSEC GT
        fields.append(f)
    gt = _gt_npz(tmp_path / "g.npz", fields)
    d = GT_DT_US
    windows = [(T0_US + d // 3, T0_US + d // 2),                 # short: scaled branch
               (T0_US, T0_US + d), (T0_US, T0_US + 2 * d),       # boundary-aligned
               (T0_US + d // 4, T0_US + 3 * d + d // 2),          # partial first and last
               (T0_US + 1, T0_US + 5 * d - 1), (T0_US + d - 1, T0_US + 2 * d + 1)]
    for a, b in windows:
        got = mvsec.gt_flow_for_window(gt, a, b)
        ref = _reference_step_by_step(gt, a, b)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])
