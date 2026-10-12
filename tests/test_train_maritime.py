"""Maritime training data (Phase 7 step 3; D-056): window rule, teacher/none GT, a 640x360
dataset item, the train-split guard, and the arm configs."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from evflow.teacher.cache import TeacherCache, write
from evflow.train.data import (
    Augment,
    Clip,
    ClipData,
    EventArrays,
    MaritimeGT,
    WindowDataset,
    WindowRef,
    batch_inputs,
    collate,
    maritime_key,
    maritime_windows,
)
from evflow.train.loop import load_config, validate

REPO = Path(__file__).resolve().parents[1]


def test_window_rule():
    np.testing.assert_array_equal(maritime_windows(15, 2), np.arange(2, 14))
    assert maritime_windows(169, 2)[-1] == 167          # window 167 spans frames 167 -> 168
    with pytest.raises(ValueError):
        maritime_windows(3, 2)


def _teacher(tmp_path, n=4, h=36, w=64):
    rng = np.random.default_rng(0)
    out = {"flow": rng.normal(0, 2, (n, 2, h, w)).astype(np.float32),
           "valid": rng.random((n, h, w)) > 0.3, "valid_literal": rng.random((n, h, w)) > 0.2}
    p = tmp_path / "t.h5"
    write(p, out, {"unit": "x/y", "frames_sha256": "abc"})
    return p, out


def test_teacher_cache_roundtrip(tmp_path):
    p, out = _teacher(tmp_path)
    with TeacherCache(p) as tc:
        assert tc.n == 4 and tc.attrs["frames_sha256"] == "abc" and tc.attrs["width"] == 64
        f, v = tc.get(2)
        np.testing.assert_array_equal(f, out["flow"][2])
        np.testing.assert_array_equal(v, out["valid"][2])
        np.testing.assert_array_equal(tc.get_literal(1), out["valid_literal"][1])
    with h5py.File(p) as f:
        np.testing.assert_array_equal(f["frame_index"][:], np.arange(4))


def test_maritime_gt_teacher_and_none(tmp_path):
    p, out = _teacher(tmp_path)
    keys = {maritime_key(3, k) for k in range(4)}
    gt = MaritimeGT(keys, {3: p}, height=36, width=64)
    assert maritime_key(3, 2) in gt and maritime_key(2, 2) not in gt
    f, v = gt.get(maritime_key(3, 2))
    np.testing.assert_array_equal(f, out["flow"][2])
    np.testing.assert_array_equal(v, out["valid"][2])
    import pickle
    gt2 = pickle.loads(pickle.dumps(gt))                # fork/spawn-safe (no open handles)
    np.testing.assert_array_equal(gt2.get(maritime_key(3, 1))[0], out["flow"][1])
    none = MaritimeGT(keys, None, height=36, width=64)
    f, v = none.get(maritime_key(3, 0))
    assert f.shape == (2, 36, 64) and not f.any() and not v.any()


def test_dataset_item_at_640x360_with_crop_and_flip():
    rng = np.random.default_rng(1)
    n = 4000
    t = np.sort(rng.integers(0, 66_667, n)).astype(np.int64)
    arr = EventArrays(rng.integers(0, 640, n).astype(np.uint16),
                      rng.integers(0, 360, n).astype(np.uint16), t,
                      rng.integers(0, 2, n).astype(np.int8), source="fake")
    keys = [maritime_key(0, k) for k in (0, 1)]
    refs = [WindowRef(0, 1, keys[0], 0, 33_333), WindowRef(0, 1, keys[1], 33_333, 66_667)]
    gt = MaritimeGT(set(keys), None)
    ds = WindowDataset([ClipData(Clip("lars_train/x", "lars_train", 0, 2), [arr])], {1: refs},
                       {1: gt}, augment=Augment(crop=(256, 256), hflip=True), hood_row=None,
                       height=360, width=640)
    torch.manual_seed(0)
    b = collate([ds[0], ds[1]])
    assert b["flow"].shape == (2, 2, 256, 256) and not b["mask"].any()
    assert bool(b["ev_cm_keep"].all())                  # no hood rule for maritime
    assert int(b["ev_x"].max()) < 256 and int(b["ev_y"].max()) < 256
    inp = batch_inputs(b, "voxel5", 256, 256, "cpu")
    assert inp.shape == (2, 5, 256, 256)


def test_arm_configs():
    for arm, loss, metric, mode in (("maritime_selfsup", "cm", "fwl", "max"),
                                    ("maritime_teacher", "supervised", "epe", "min")):
        c = load_config(REPO / "configs" / f"train_{arm}.yaml")
        validate(c)
        assert c["arm"] == arm and c["loss"] == loss and c["events"]["train"] == "maritime"
        assert c["selection"] == {"split": "mvsec_tune", "metric": metric, "mode": mode, "dt": 1}
        assert c["events"]["tune_select"] == "v2e"
        assert c["maritime"]["corpora"] == ["smd_train", "lars_train"]
        assert c["train"]["dts"] == [1]
    assert load_config(REPO / "configs" / "train_maritime_selfsup.yaml")["losses"]["cm"][
        "min_count"] == 0.01
    assert "min_count" not in load_config(REPO / "configs" / "train_v2e_selfsup.yaml")[
        "losses"]["cm"]                                  # Phase 5 arms unchanged


@pytest.mark.dataset
def test_load_maritime_on_the_corpus():
    from evflow.train.data import load_maritime
    ds, prov = load_maritime(supervision="none", warmup=2, augment=None,
                             max_train_windows=50)
    assert prov["n_units"] == 118 and len(ds) == 50
    assert all(s["split"] in ("smd_train", "lars_train") for s in prov["stores"].values())
    it = ds[0]
    assert it["flow"].shape == (2, 360, 640)
