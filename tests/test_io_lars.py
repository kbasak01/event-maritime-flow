"""LaRS reader on a fake LaRS layout (Phase 7; D-024, D-054)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from evflow.io import frames, lars

from fixtures.fake_sources import write_fake_lars

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake(tmp_path):
    return tmp_path, write_fake_lars(tmp_path)


def test_clip_frames_and_keyframe(fake):
    root, info = fake
    c = info["clip"]
    files = lars.clip_frames(root, "val", c["prefix"], c["first"], c["last"])
    assert len(files) == c["n_frames"] and files[0].name.endswith(f"_{c['first']:05d}.jpg")
    assert lars.keyframe_indices(c) == [9]
    assert lars.clip_spec(c) == f"{c['prefix']}:{c['first']}:{c['last']}"
    with pytest.raises(FileNotFoundError):
        lars.clip_frames(root, "val", c["prefix"], c["first"], c["last"] + 1)


def test_keyframe_outside_clip_is_refused(fake):
    c = dict(fake[1]["clip"], keyframes=["fake_seq_01_00099"])
    with pytest.raises(ValueError):
        lars.keyframe_indices(c)


def test_every_kth_rule():
    clips = [{"clip": str(i)} for i in range(197)]
    got = lars.every_kth(clips, 40)
    assert [c["clip"] for c in got] == [str(i) for i in range(0, 160, 4)]
    assert len(lars.every_kth(clips * 13 + clips[:42], 110)) == 110   # 2603 // 110 = 23
    with pytest.raises(ValueError):
        lars.every_kth(clips, 0)


def test_semantic_mask_and_classes(fake):
    root, info = fake
    lab = lars.semantic_mask(root, "val", info["keyframe"])
    np.testing.assert_array_equal(lab, info["label"])
    m = lars.class_masks(lab)
    assert set(m) == {"water", "sky", "obstacle", "ignore"}
    total = sum(m[c].astype(int) for c in m)
    assert (total == 1).all()                       # a partition
    assert m["ignore"].sum() == 6 and m["obstacle"].sum() == 4 * 32
    # nearest resize keeps labels (no blending into new values), geometry stretched
    small = lars.semantic_mask(root, "val", info["keyframe"], size=(40, 30))
    assert small.shape == (30, 40) and set(np.unique(small)) <= {0, 1, 2, 255}


def test_verify_class_mapping(fake, tmp_path):
    root, info = fake
    rep = lars.verify_class_mapping(root, "val")
    assert rep["n_keyframes"] == 1 and rep["mapping"] == {"obstacle": 0, "water": 1, "sky": 2}
    # swap water and sky areas in the panoptic file -> mismatch is detected
    p = root / "val" / "panoptic_annotations.json"
    doc = json.loads(p.read_text())
    seg = doc["annotations"][0]["segments_info"]
    seg[1]["area"], seg[2]["area"] = seg[2]["area"], seg[1]["area"]
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        lars.verify_class_mapping(root, "val")


def test_attributes_flattened_and_null_kept(fake):
    root, info = fake
    att = lars.load_attributes(root, "val")[info["keyframe"]]
    assert att["waves"] == "moderate" and att["reflections"] is None
    assert att["special.glitter"] is True and att["special.rain"] is False
    assert set(att) == set(lars.ATTRIBUTE_FIELDS) | {f"special.{k}" for k in lars.SPECIAL_FIELDS}


def test_frame_clock_and_geometry():
    t = frames.frame_times_us(4, 30.0, t_offset_us=1000)
    np.testing.assert_array_equal(t, [1000, 34333, 67667, 101000])
    assert frames.scale_factors((1278, 958), (640, 360)) == (640 / 1278, 360 / 958)
    im = np.zeros((958, 1278, 3), np.uint8)
    assert frames.resize_like_v2e(im, (640, 360)).shape == (360, 640, 3)


def test_assumed_fps_is_configured():
    doc = yaml.safe_load((REPO / "configs" / "v2e_profiles.yaml").read_text())
    assert float(doc["lars"]["assumed_fps"]) == 30.0


@pytest.mark.dataset
def test_real_lars_mapping_and_attributes():
    paths = yaml.safe_load((REPO / "configs" / "paths.local.yaml").read_text())
    root = paths["lars"]["root"]
    for subset in ("val", "train"):
        rep = lars.verify_class_mapping(root, subset)
        assert max(rep["max_abs_area_diff_fraction"].values()) == 0.0
    att = lars.load_attributes(root, "val")
    assert len(att) == 198
    assert {a["waves"] for a in att.values()} == {"still", "moderate", "rough"}
