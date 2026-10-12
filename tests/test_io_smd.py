"""SMD reader on a tiny AVI (Phase 7; D-025, D-054)."""

import numpy as np
import pytest

from evflow.io import smd

from fixtures.fake_sources import write_fake_smd


def test_segment_rule():
    # train: [0, 5.6] s of a 299-frame 30 fps video -> frames 0..168
    s = smd.segment(0.0, 5.6, 30.0, 299)
    assert (s["first_frame"], s["last_frame"], s["n_frames"]) == (0, 168, 169)
    assert s["video_s"] == 5.6
    # test: [0, 10] s of a 279-frame video is cut at its last frame
    s = smd.segment(0.0, 10.0, 30.0, 279)
    assert (s["first_frame"], s["last_frame"]) == (0, 278)
    # the runner recomputes round(stop * fps) from the returned stop_s
    assert int(round(s["stop_s"] * 30.0)) == 278
    with pytest.raises(ValueError):
        smd.segment(5.0, 5.0, 30.0, 299)


def test_probe_and_read(tmp_path):
    v = tmp_path / "MVI_0000_VIS_OB.avi"
    write_fake_smd(v, n_frames=20)
    info = smd.probe(v)
    assert info["fps"] == 30.0 and info["n_frames"] == 20 and info["width"] == 64
    fr = smd.read_frames(v, 3, 6)
    assert len(fr) == 4 and fr[0].shape == (36, 64, 3)
    np.testing.assert_allclose([f.mean() for f in fr], [30, 40, 50, 60], atol=3)
    small = smd.read_frames(v, 0, 1, size=(32, 18))
    assert small[0].shape == (18, 32, 3)
    with pytest.raises(ValueError):
        smd.read_frames(v, 18, 25)
    assert smd.video_path("/x", "MVI_0000_VIS_OB").name == "MVI_0000_VIS_OB.avi"
