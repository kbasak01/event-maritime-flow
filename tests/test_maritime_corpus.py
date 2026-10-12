"""Maritime corpus unit rules (Phase 7 step 1; D-026, D-054). Pure functions, no v2e."""

from pathlib import Path

import pytest
import yaml

from evflow.sim import maritime_corpus as mc

REPO = Path(__file__).resolve().parents[1]


def _clips(n, subset_prefix="c"):
    return [{"clip": f"{subset_prefix}{i}_{10 * i:05d}", "prefix": f"{subset_prefix}{i}",
             "first": 10 * i, "last": 10 * i + 14, "n_frames": 15,
             "keyframes": [f"{subset_prefix}{i}_{10 * i + 9:05d}"]} for i in range(n)]


@pytest.fixture
def inputs():
    cfg = yaml.safe_load((REPO / "configs" / "v2e_corpus_maritime.yaml").read_text())
    prof = yaml.safe_load((REPO / "configs" / "v2e_profiles.yaml").read_text())
    splits = {
        "smd_train": {"split": "smd_train", "videos": [f"T{i}" for i in range(8)]},
        "smd_test": {"split": "smd_test", "videos": ["A", "B", "C"]},
        "lars_train": {"split": "lars_train", "subset": "train", "clips": _clips(2603)},
        "lars_eval": {"split": "lars_eval", "subset": "val", "clips": _clips(197, "v")},
    }
    probe = {v: {"fps": 30.0, "n_frames": 299} for v in splits["smd_train"]["videos"]}
    probe.update({"A": {"fps": 30.0, "n_frames": 279}, "B": {"fps": 30.0, "n_frames": 600},
                  "C": {"fps": 30.0, "n_frames": 600}})
    return cfg, prof, splits, probe


def test_plan_counts_and_segments(inputs):
    units = mc.plan_units(*inputs)
    by = {}
    for u in units:
        by.setdefault(u["corpus"], []).append(u)
    assert {k: len(v) for k, v in by.items()} == {
        "trial": 2, "smd_train": 8, "smd_test": 3, "lars_train": 110,
        "lars_eval_ct067": 40, "lars_eval_ct100": 40, "lars_eval_ct150": 40}
    assert units[0]["corpus"] == "trial" and units[1]["corpus"] == "trial"
    assert all(u["frames"] == [0, 168] and u["video_s"] == 5.6 for u in by["smd_train"])
    assert [u["frames"][1] for u in by["smd_test"]] == [278, 300, 300]
    assert len({u["key"] for u in units}) == len(units)
    assert all(u["seed"] == 1337 and u["profile"] == "maritime_640x360" for u in units)


def test_lars_clip_choice_every_kth(inputs):
    units = mc.plan_units(*inputs)
    tr = [u for u in units if u["corpus"] == "lars_train"]
    assert tr[0]["clip"] == "c0:0:14" and tr[1]["clip"] == "c23:230:244"       # k = 23
    ev = [u for u in units if u["corpus"] == "lars_eval_ct100"]
    assert [u["clip"].split(":")[0] for u in ev[:3]] == ["v0", "v4", "v8"]      # k = 4
    # the same 40 clips in every CT variant
    for tag in ("ct067", "ct150"):
        other = [u["clip"] for u in units if u["corpus"] == f"lars_eval_{tag}"]
        assert other == [u["clip"] for u in ev]
    assert all(u["keyframes_local"] == [9] and u["split"] == "lars_eval" for u in ev)
    assert {u["subset"] for u in ev} == {"val"} and {u["subset"] for u in tr} == {"train"}


def test_ct_overrides_scale_thresholds_and_sigma(inputs):
    units = mc.plan_units(*inputs)
    ov = {u["corpus"]: u["overrides"] for u in units if u["corpus"].startswith("lars_eval")}
    assert ov["lars_eval_ct100"] == {}
    lo, hi = ov["lars_eval_ct067"], ov["lars_eval_ct150"]
    assert float(lo["--pos_thres"]) == float(lo["--neg_thres"]) == pytest.approx(0.6 * 0.67)
    assert float(hi["--sigma_thres"]) == pytest.approx(0.04 * 1.5)
    assert mc.ct_tag(0.67) == "ct067" and mc.ct_tag(1.5) == "ct150"
    with pytest.raises(ValueError):
        mc.ct_variant_overrides({"--pos_thres": "0.5", "--neg_thres": "0.6",
                                 "--sigma_thres": "0.04"}, 1.5)


def test_unit_argv(inputs, tmp_path):
    units = mc.plan_units(*inputs)
    smd = next(u for u in units if u["corpus"] == "smd_test")
    a = mc.unit_argv(smd, tmp_path)
    assert a[:3] == ["smd", "--video", "A.avi"]
    stop = float(a[a.index("--stop") + 1])
    assert int(round(stop * 30.0)) == smd["frames"][1]          # runner reproduces the frames
    lars = next(u for u in units if u["corpus"] == "lars_eval_ct150")
    a = mc.unit_argv(lars, tmp_path)
    assert a[:5] == ["lars", "--subset", "val", "--clip", lars["clip"]]
    assert "--set=--pos_thres=0.9" in a
    assert a[a.index("--out") + 1] == str(tmp_path / "lars_eval_ct150")
