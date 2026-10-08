"""DSEC converter and flow-GT reader on fake files with the observed layout (D-016, D-017)."""

import h5py
import numpy as np
import pytest

from evflow.io import dsec
from evflow.io.store import EventSlicer, EventStore

from fixtures.fake_sources import write_fake_dsec


@pytest.fixture
def fake(tmp_path):
    seq = tmp_path / "zurich_fake"
    info = write_fake_dsec(seq)
    return seq, info


def test_convert_events_copies_and_casts(fake, tmp_path):
    seq, info = fake
    dst = tmp_path / "out.h5"
    summary = dsec.convert_events(seq, dst, sequence="zurich_fake", chunk=3001)
    assert summary["n_events"] == len(info["t"])
    with EventStore(dst) as s:
        assert s.t_offset == info["t_offset"]
        assert s.attrs["distortion"] == "raw"
        assert s.attrs["polarity_mapping"] and s.attrs["time_conversion"]
        assert s.attrs["source"] == "dsec/zurich_fake/events_left"
        assert (s.width, s.height) == (640, 480)
        np.testing.assert_array_equal(s.events["t"][:], info["t"].astype(np.int64))
        np.testing.assert_array_equal(s.events["p"][:], info["p"].astype(np.int8))
        np.testing.assert_array_equal(s.events["x"][:], info["x"])
        np.testing.assert_array_equal(s.ms_to_idx, info["ms_to_idx"].astype(np.int64))
        np.testing.assert_array_equal(s.rectify_map, info["rectify_map"])


def test_convert_detects_ms_to_idx_mismatch(fake, tmp_path):
    seq, _ = fake
    with h5py.File(seq / "events_left" / "events.h5", "r+") as f:
        f["ms_to_idx"][5] += 1
    with pytest.raises(ValueError, match="ms_to_idx"):
        dsec.convert_events(seq, tmp_path / "o.h5", sequence="x")


def test_rectified_slice_matches_map(fake, tmp_path):
    seq, info = fake
    dst = tmp_path / "out.h5"
    dsec.convert_events(seq, dst, sequence="zurich_fake")
    with EventSlicer(dst) as sl:
        t0 = info["t_offset"] + 100_000
        raw = sl.get_events(t0, t0 + 50_000, rectify=False)
        rect = sl.get_events(t0, t0 + 50_000, rectify=True)
        default = sl.get_events(t0, t0 + 50_000)  # store has a map -> rectified by default
    np.testing.assert_array_equal(default["x"], rect["x"])
    rmap = info["rectify_map"]
    np.testing.assert_allclose(rect["x"], rmap[raw["y"], raw["x"], 0])
    np.testing.assert_allclose(rect["y"], rmap[raw["y"], raw["x"], 1])


def test_flow_gt_index_and_decode(fake):
    seq, info = fake
    gt = dsec.FlowGT(seq)
    assert len(gt) == 3
    for k, (a, b) in enumerate(info["flow_rows"]):
        flow, valid, (fa, fb) = gt[k]
        assert (fa, fb) == (a, b)
        want_flow, want_valid = info["flows"][k]
        np.testing.assert_array_equal(valid, want_valid)
        np.testing.assert_allclose(flow[:, valid], want_flow[:, valid])


def test_flow_gt_png_name_from_image_index(fake):
    seq, _ = fake
    gt = dsec.FlowGT(seq)
    assert [p.name for p in gt.png_paths] == ["000002.png", "000004.png", "000006.png"]
