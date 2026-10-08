"""Converted real stores vs their sources (marker ``dataset``; run with ``make test-data``).

Needs configs/paths.local.yaml and the stores written by
    python scripts/convert_events.py mvsec --sequence indoor_flying1
    python scripts/convert_events.py dsec  --sequence thun_00_a
"""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest
import yaml

from evflow.io import dsec, mvsec
from evflow.io.store import REQUIRED_ATTRS, EventSlicer, EventStore, compute_ms_to_idx

pytestmark = pytest.mark.dataset

REPO = Path(__file__).resolve().parents[1]
CHUNK = 20_000_000


@pytest.fixture(scope="module")
def cfg():
    p = REPO / "configs/paths.local.yaml"
    if not p.exists():
        pytest.skip("configs/paths.local.yaml missing")
    return yaml.safe_load(p.read_text())


def _store(cfg, name):
    path = Path(cfg["outputs"]["events"]) / name
    if not path.exists():
        pytest.fail(f"{path} missing: run scripts/convert_events.py first")
    return path


def _check_integrity(path, rng):
    """Checklist A1 on a canonical store; returns the full t array."""
    with EventStore(path) as s:
        for a in REQUIRED_ATTRS:
            assert a in s.attrs, a
        assert s.attrs["polarity_convention"] == "0=OFF,1=ON"
        ev = s.events
        assert ev["t"].dtype == np.int64 and ev["p"].dtype == np.int8
        assert ev["x"].dtype == np.uint16 and ev["y"].dtype == np.uint16
        t = np.empty(s.n_events, np.int64)
        for lo in range(0, s.n_events, CHUNK):
            hi = min(lo + CHUNK, s.n_events)
            t[lo:hi] = ev["t"][lo:hi]
            p = ev["p"][lo:hi]
            assert np.isin(p, (0, 1)).all()
            assert int(ev["x"][lo:hi].max()) < s.width and int(ev["y"][lo:hi].max()) < s.height
        assert t[0] >= 0 and np.all(np.diff(t) >= 0)
        np.testing.assert_array_equal(s.ms_to_idx, compute_ms_to_idx(t))
        m = s.ms_to_idx
        for ms in [0, 1, len(m) - 2, len(m) - 1, *rng.integers(0, len(m), 20)]:
            i = int(m[ms])
            if i < len(t):
                assert t[i] >= ms * 1000
            if i > 0:
                assert t[i - 1] < ms * 1000
        assert m[-1] == len(t)
    return t


def test_mvsec_indoor_flying1_store(cfg):
    rng = np.random.default_rng(20261008)
    path = _store(cfg, "mvsec_indoor_flying1_left.h5")
    t = _check_integrity(path, rng)
    src = Path(cfg["mvsec"]["root"]) / "indoor_flying" / "indoor_flying1_data.hdf5"
    with h5py.File(src, "r") as f, EventStore(path) as s:
        e = f["davis/left/events"]
        assert e.shape[0] == s.n_events
        idx = np.sort(rng.choice(s.n_events, 5000, replace=False))
        rows = e[idx]
        assert s.attrs["distortion"] == "raw" and (s.width, s.height) == (346, 260)
        np.testing.assert_array_equal(s.events["x"][idx], rows[:, 0].astype(np.uint16))
        np.testing.assert_array_equal(s.events["y"][idx], rows[:, 1].astype(np.uint16))
        np.testing.assert_array_equal(s.events["p"][idx], (rows[:, 3] > 0).astype(np.int8))
        np.testing.assert_array_equal(t[idx] + s.t_offset, np.round(rows[:, 2] * 1e6))


def test_mvsec_gt_same_pixel_space_and_window(cfg):
    gt = mvsec.GTFlow(Path(cfg["mvsec"]["gt_flow"]) / "indoor_flying" /
                      "indoor_flying1_gt_flow_dist.npz")
    assert gt.memmapped and gt.shape == (260, 346)  # same grid as davis/left/events (A5)
    src = Path(cfg["mvsec"]["root"]) / "indoor_flying" / "indoor_flying1_data.hdf5"
    aps = mvsec.read_aps_timestamps_us(src)
    k = int(np.searchsorted(aps, gt.timestamps_us[0])) + 600  # mid-flight APS frame
    for dt in (1, 4):
        flow, valid = mvsec.gt_flow_for_window(gt, int(aps[k]), int(aps[k + dt]))
        assert valid.mean() > 0.3 and np.isfinite(flow).all()
        if dt == 1:
            mag1 = np.median(np.hypot(*flow)[valid])
    assert np.median(np.hypot(*flow)[valid]) > 2 * mag1  # dt=4 GT is larger, not identical
    with EventSlicer(_store(cfg, "mvsec_indoor_flying1_left.h5")) as sl:
        ev = sl.get_events(int(aps[k]), int(aps[k + 1]))
    hit = np.zeros((260, 346), bool)
    hit[ev["y"], ev["x"]] = True
    assert (hit & valid).sum() > 100  # events and GT overlap on the same grid


def test_dsec_thun_00_a_store(cfg):
    rng = np.random.default_rng(20261008)
    path = _store(cfg, "dsec_thun_00_a_left.h5")
    t = _check_integrity(path, rng)
    ev_dir = Path(cfg["dsec"]["root"]) / "thun_00_a" / "events_left"
    with h5py.File(ev_dir / "events.h5", "r") as f, EventStore(path) as s:
        assert f["events/t"].shape[0] == s.n_events
        assert int(f["t_offset"][()]) == s.t_offset
        np.testing.assert_array_equal(s.ms_to_idx, np.asarray(f["ms_to_idx"], np.int64))
        lo = int(rng.integers(0, s.n_events - 100_000))
        for k in ("x", "y", "p"):
            np.testing.assert_array_equal(s.events[k][lo : lo + 100_000],
                                          f[f"events/{k}"][lo : lo + 100_000])
        np.testing.assert_array_equal(t[lo : lo + 100_000], f["events/t"][lo : lo + 100_000])
        with h5py.File(ev_dir / "rectify_map.h5", "r") as r:
            np.testing.assert_array_equal(s.rectify_map, r["rectify_map"][:])
        assert s.attrs["distortion"] == "raw"


def test_dsec_flow_gt_and_rectified_window(cfg):
    seq = Path(cfg["dsec"]["root"]) / "thun_00_a"
    gt = dsec.FlowGT(seq)
    assert len(gt) == 41
    flow, valid, (a, b) = gt[0]
    assert flow.shape == (2, 480, 640) and 0.05 < valid.mean() < 1.0
    assert np.abs(flow[:, valid]).max() < 256
    assert 99_000 < b - a < 101_000
    with EventSlicer(_store(cfg, "dsec_thun_00_a_left.h5")) as sl:
        raw = sl.get_events(a, b, rectify=False)
        ev = sl.get_events(a, b, rectify=True)
    assert ev is not None and len(ev["t"]) > 1000
    # Rectification only pushes events near the raw border out of the image (the map sends no
    # pixel farther than 14 px from the border outside; D-016), and keeps the centre in place.
    inside = (ev["x"] >= 0) & (ev["x"] <= 639) & (ev["y"] >= 0) & (ev["y"] <= 479)
    x, y = raw["x"].astype(int), raw["y"].astype(int)
    border = np.minimum.reduce([x, 639 - x, y, 479 - y])
    assert border[~inside].max() <= 15
    centre = (np.abs(x - 320) < 50) & (np.abs(y - 240) < 50)
    assert np.all(np.hypot(ev["x"][centre] - x[centre], ev["y"][centre] - y[centre]) < 10)
