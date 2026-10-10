"""scripts/summarise_throughput.py: v2e log parsing, budget costing, concurrency factor."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("st", REPO / "scripts/summarise_throughput.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

LOG = ("INFO SloMo average upsampling factor=31.00; average DVS timestamp resolution=737us\r"
       "slomo-interp:  50%|#####     | 14/28 [00:40<00:40,  2.9s/ batch of 8 fr]\r"
       "slomo-interp: 100%|##########| 28/28 [01:21<00:00,  2.9s/ batch of 8 fr]\n"
       "dvs: 100%|##########| 6789/6789 [01:05<00:00, 104.2fr/s]\n")


def test_stage_times():
    s = st.stage_times(LOG)
    assert s == {"slomo_s": 81, "dvs_s": 65, "upsampling_factor": 31.0}
    assert st.stage_times("nothing") == {"slomo_s": None, "dvs_s": None,
                                         "upsampling_factor": None}


ROWS = {"a": {"overhead_s": 10.0, "marginal_per_video_s": 30.0, "wall_s": 40.0,
              "wall_per_video_s": 32.0},
        "lars_0": {"overhead_s": 5.0, "marginal_per_video_s": 80.0, "wall_s": 30.0},
        "lars_1": {"overhead_s": 5.0, "marginal_per_video_s": 80.0, "wall_s": 50.0}}


def test_cost_item_video_and_clips():
    v = st.cost_item({"name": "x", "video_s": 120.0, "units": 2, "rate_from": "a"}, ROWS)
    assert v["gpu_hours"] == round((120 * 30 + 2 * 10) / 3600, 3)
    c = st.cost_item({"name": "y", "clips": 10, "multiplier": 3, "rate_from": "lars_.*"}, ROWS)
    assert c["gpu_hours"] == round(10 * 40.0 * 3 / 3600, 3)
    with pytest.raises(KeyError):
        st.cost_item({"name": "z", "video_s": 1, "rate_from": "nope"}, ROWS)


def test_concurrency_factor():
    conc = [{"concurrency": 4, "effective_gpu_s_per_video_s": 16.0}]
    assert st.concurrency_factor(conc, ROWS, 1, "a") == 1.0
    assert st.concurrency_factor(conc, ROWS, 4, "a") == 0.5
    with pytest.raises(KeyError):
        st.concurrency_factor(conc, ROWS, 3, "a")


def test_measured_item_from_corpus_units(tmp_path, monkeypatch):
    doc = {"units": [{"corpus": "trial", "wall_time_s": 100.0},
                     {"corpus": "tune", "wall_time_s": 7200.0},
                     {"corpus": "rand0", "wall_time_s": 7200.0}]}
    (tmp_path / "c.json").write_text(json.dumps(doc))
    monkeypatch.setattr(st, "REPO", tmp_path)
    one = st.measured_item({"measured_from": "c.json", "concurrency": 1, "corpora": ["trial"]})
    assert one["units"] == 1 and one["gpu_hours"] == round(100 / 3600, 3)
    four = st.measured_item({"measured_from": "c.json", "concurrency": 4,
                             "corpora": ["tune", "rand0"]})
    assert four["units"] == 2 and four["gpu_hours"] == 1.0


def test_measured_item_grid_filter(tmp_path, monkeypatch):
    doc = {"cells": [{"grid": "calibration_grid", "units": [{"wall_time_s": 3600.0}]},
                     {"grid": "calibration_grid_timing", "units": [{"wall_time_s": 7200.0}]}]}
    (tmp_path / "t.json").write_text(json.dumps(doc))
    monkeypatch.setattr(st, "REPO", tmp_path)
    it = st.measured_item({"measured_from": "t.json", "concurrency": 1,
                           "grid": "calibration_grid_timing"})
    assert it["units"] == 1 and it["gpu_hours"] == 2.0


def test_lost_unit_minutes_item():
    it = st.cost_item({"name": "x", "lost_unit_minutes": 120, "concurrency": 4}, {})
    assert it["gpu_hours"] == 0.5 and it["measured"]
