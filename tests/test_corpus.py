"""sim/corpus + sim/splits rules on synthetic splits and clocks (no data), plus one dataset check
that the real plan reproduces the D-026 clip frames."""

import numpy as np
import pytest

from evflow.sim import corpus as K
from evflow.sim import splits as SP

S = 1_000_000
AP = 25_000                                   # 40 fps synthetic APS clock
CFG = {
    "warmup_frames": 10, "max_unit_s": 60.0, "max_dt": 4,
    "cal": {"profile": "mvsec_cal", "seed": 1337},
    "tune": {"split": "mvsec_tune"}, "test": {"split": "mvsec_test"},
    "train": {"split": "mvsec_train", "range_index": 1, "n_clips": 4, "clip_s": 60.0},
    "rand": {"profile": "mvsec_rand", "draw_seed": 20261008, "seed_base": 2000},
    "trials": {"split": "mvsec_tune", "n_frames": 200, "extremes": ["lo", "hi"]},
}
PROF = {"profiles": {"mvsec_rand": {"randomise": {"thres_range": [0.28, 0.52],
                                                  "sigma_range": [0.02, 0.052],
                                                  "corpora": 4}}}}


def _aps(n, t0=1_000 * S):
    return t0 + np.arange(n, dtype=np.int64) * AP


def _splits(aps_train, aps_test):
    def frames(a, b):
        i0, i1 = K.frames_in(aps_train, a, b)
        return {"aps_first": i0, "aps_last": i1}
    t0 = int(aps_train[0])
    tune_r = [t0 + 100 * S, t0 + 160 * S]
    train_r = [[t0, t0 + 70 * S], [t0 + 190 * S, t0 + 600 * S]]
    test_seq = {"sequence": "indoor_flying1", "windows": {
        "dt1": {"aps_first": 9, "aps_last": len(aps_test) - 2},
        "dt4": {"aps_first": 9, "aps_last": len(aps_test) - 5}}}
    return {
        "mvsec_tune": {"split": "mvsec_tune", "dataset": "mvsec", "sequence": "outdoor_day2",
                       "ranges_us": [tune_r], "aps": [frames(*tune_r)]},
        "mvsec_train": {"split": "mvsec_train", "dataset": "mvsec", "sequence": "outdoor_day2",
                        "ranges_us": train_r, "aps": [frames(*r) for r in train_r]},
        "mvsec_test": {"split": "mvsec_test", "dataset": "mvsec", "sequences": [test_seq]},
    }


@pytest.fixture
def setup():
    aps_train, aps_test = _aps(30_000), _aps(3_500, 5_000 * S)     # test: 87.5 s
    splits = _splits(aps_train, aps_test)
    aps = {"outdoor_day2": aps_train, "indoor_flying1": aps_test}
    return splits, aps


def test_plan_deterministic_and_complete(setup):
    splits, aps = setup
    a = K.plan_units(CFG, PROF, splits, aps)
    assert a == K.plan_units(CFG, PROF, splits, aps)
    by = {}
    for u in a:
        by.setdefault(u["corpus"], []).append(u)
    assert [len(by[c]) for c in ("trial", "tune", "test", "train_cal")] == [2, 1, 2, 4]
    assert all(len(by[f"rand{k}"]) == 4 for k in range(4))
    assert all(u["video_s"] <= 60.0 for u in a)


def test_train_clips_rule(setup):
    splits, aps = setup
    clips = K.train_clips(splits["mvsec_train"], aps["outdoor_day2"], 4, 60.0, 1)
    r0, r1 = splits["mvsec_train"]["ranges_us"][1]
    starts = [int(round(t)) for t in np.linspace(r0, r1 - 60 * S, 4)]
    for (f0, f1), s in zip(clips, starts, strict=True):
        assert aps["outdoor_day2"][f0] >= s > aps["outdoor_day2"][f0 - 1]
        assert aps["outdoor_day2"][f1] < s + 60 * S <= aps["outdoor_day2"][f1 + 1]


def test_every_test_window_owned_once(setup):
    splits, aps = setup
    seq = splits["mvsec_test"]["sequences"][0]
    chunks = K.chunk_test_sequence(seq, aps["indoor_flying1"], 10, 60.0, 4)
    assert len(chunks) == 2
    owned = np.concatenate([np.arange(c["owned"][0], c["owned"][1] + 1) for c in chunks])
    for dt in (1, 4):
        w = seq["windows"][f"dt{dt}"]
        starts = np.arange(w["aps_first"], w["aps_last"] + 1)
        assert np.array_equal(np.unique(owned), owned)            # no window owned twice
        assert np.isin(starts, owned).all()
        for c in chunks:                                          # owner holds every frame
            k = starts[(starts >= c["owned"][0]) & (starts <= c["owned"][1])]
            assert c["aps"][0] <= k.min() and k.max() + dt <= c["aps"][1]
    assert chunks[1]["aps"][0] == chunks[1]["owned"][0] - 10      # warm-up overlap
    assert chunks[0]["lead_frames"] == 9 and chunks[1]["lead_frames"] == 0


def test_rand_draws_inside_frozen_range():
    d = K.rand_draws(PROF["profiles"]["mvsec_rand"]["randomise"], 20261008, 4)
    assert len(d) == 16 and d == K.rand_draws(PROF["profiles"]["mvsec_rand"]["randomise"],
                                              20261008, 4)
    assert all(0.28 <= x["pos_neg_thres"] <= 0.52 and 0.02 <= x["sigma_thres"] <= 0.052
               for x in d)
    assert len({x["pos_neg_thres"] for x in d}) == 16


def test_rand_units_new_sensor_and_overrides(setup):
    splits, aps = setup
    units = K.plan_units(CFG, PROF, splits, aps)
    cal = [u for u in units if u["profile"] == "mvsec_cal"]
    rnd = [u for u in units if u["corpus"].startswith("rand")]
    assert {u["seed"] for u in cal} == {1337}
    assert len({u["seed"] for u in rnd}) == 16
    for u in rnd:
        assert u["overrides"]["--pos_thres"] == u["overrides"]["--neg_thres"]
    lo = next(u for u in units if u["name"] == "trial_rand_lo")
    assert lo["overrides"] == {"--pos_thres": "0.28", "--neg_thres": "0.28",
                               "--sigma_thres": "0.052"}


def test_guard_rejects_test_frames_for_training(setup):
    splits, _ = setup
    with pytest.raises(SP.LeakageError):
        SP.check_aps_in_split(splits["mvsec_train"], "indoor_flying1", 10, 100)
    with pytest.raises(SP.LeakageError):          # tune frames claimed as train
        a = splits["mvsec_tune"]["aps"][0]
        SP.check_aps_in_split(splits["mvsec_train"], "outdoor_day2", a["aps_first"],
                              a["aps_first"] + 50)
    with pytest.raises(SP.LeakageError):          # lead frames only for evaluation splits
        a = splits["mvsec_tune"]["aps"][0]
        SP.check_aps_in_split(splits["mvsec_tune"], "outdoor_day2", a["aps_first"] - 5,
                              a["aps_last"], lead_frames=5)
    with pytest.raises(SP.LeakageError):          # past the last window's frames
        SP.check_aps_in_split(splits["mvsec_test"], "indoor_flying1", 9, 3_500)
    with pytest.raises(SP.LeakageError):
        SP.check_aps_in_split(splits["mvsec_test"], "indoor_flying1", 0, 100, lead_frames=11)


def test_guard_accepts_valid_ranges(setup):
    splits, _ = setup
    a = splits["mvsec_tune"]["aps"][0]
    SP.check_aps_in_split(splits["mvsec_tune"], "outdoor_day2", a["aps_first"], a["aps_last"])
    SP.check_aps_in_split(splits["mvsec_test"], "indoor_flying1", 0, 3_499, lead_frames=9)


def test_guard_on_committed_splits():
    test = SP.load_split("mvsec_test")
    assert SP.split_aps_bounds(test, "outdoor_day1") == [(10164, 10961)]
    with pytest.raises(SP.LeakageError):
        SP.check_aps_in_split(SP.load_split("mvsec_train"), "outdoor_day2", 3651, 6275)


@pytest.mark.dataset
def test_real_plan_matches_d026():
    plan = K.build_plan()
    clips = [u["aps"] for u in plan["units"] if u["corpus"] == "train_cal"]
    assert clips == [[7588, 10212], [13712, 16336], [19837, 22460], [25961, 28582]]
    tune = [u for u in plan["units"] if u["corpus"] == "tune"]
    assert [u["aps"] for u in tune] == [[3651, 6275]]
    test = [u for u in plan["units"] if u["corpus"] == "test"]
    assert len(test) == 7 and all(u["video_s"] <= 60 for u in plan["units"])
