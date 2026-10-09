"""sim/calibrate rules: grid order, overrides, tune-only guard, windows, randomisation range."""

import numpy as np
import pytest

from evflow.sim import calibrate as C

GRID = {"pos_neg_thres": [0.15, 0.2, 0.3, 0.4], "sigma_thres": [0.02, 0.04],
        "cutoff_hz": [0, 60], "shot_noise_rate_hz": [0, 1.0]}
TUNE = {"split": "mvsec_tune", "ranges_us": [[1_000_000, 61_000_000]],
        "aps": [{"aps_first": 100, "aps_last": 2724}]}


def test_grid_cells_order_and_count():
    cells = C.grid_cells(GRID)
    assert len(cells) == 32 and cells[0]["cell"] == "c00" and cells[-1]["cell"] == "c31"
    assert cells[0] == {"cell": "c00", "pos_neg_thres": 0.15, "sigma_thres": 0.02,
                        "cutoff_hz": 0.0, "shot_noise_rate_hz": 0.0}
    assert cells[1]["shot_noise_rate_hz"] == 1.0 and cells[8]["pos_neg_thres"] == 0.2


def test_cell_overrides_sets_both_thresholds():
    o = C.cell_overrides(C.grid_cells(GRID)[9])
    assert o["--pos_thres"] == o["--neg_thres"] == "0.2"
    assert set(o) == {"--pos_thres", "--neg_thres", "--sigma_thres", "--cutoff_hz",
                      "--shot_noise_rate_hz"}


def test_segments_spread_and_inside():
    segs = C.segments(TUNE, 3, 220)
    assert segs[0] == (100, 319) and segs[-1] == (2505, 2724)
    assert all(b - a + 1 == 220 for a, b in segs)


def test_guard_rejects_outside_tune():
    with pytest.raises(C.LeakageError):
        C.check_inside_tune(TUNE, [(50, 200)])
    with pytest.raises(C.LeakageError):
        C.check_inside_tune(TUNE, [(2600, 2800)])
    with pytest.raises(C.LeakageError):
        C.check_windows_inside_tune(TUNE, np.array([[500_000, 600_000]]))
    with pytest.raises(C.LeakageError):
        C.tune_aps_range({**TUNE, "split": "mvsec_test"})
    with pytest.raises(ValueError):
        C.segments(TUNE, 20, 220)


def test_stat_windows_warmup_and_blocks():
    aps = np.arange(0, 3000, dtype=np.int64) * 25_000
    wins, blocks = C.stat_windows(aps, (100, 319), warmup=10, block_us=500_000)
    assert len(wins) == 319 - 110 and wins[0, 0] == aps[110] and wins[-1, 1] == aps[319]
    assert np.all(np.diff(blocks[:, 0]) == 500_000) and blocks[-1, 1] <= aps[319]


def test_rand_range_rule():
    cells = C.grid_cells(GRID)
    objs = {c["cell"]: 10.0 for c in cells}
    objs["c09"] = 1.0                                # thres 0.2, sigma 0.02 (argmin)
    objs["c20"] = 1.4                                # thres 0.3, sigma 0.04 -> within 1.5x
    r = C.rand_range(cells, objs, "c09", factor=1.5, min_halfwidth=0.3)
    assert r["near_cells"] == ["c09", "c20"]
    assert r["thres_range_near"] == [0.2, 0.3]
    assert r["thres_range"] == [0.14, 0.3]           # widened below to 0.2*0.7, above already 0.3
    assert r["sigma_range"] == [0.014, 0.04]


def test_rand_range_widened_when_alone():
    cells = C.grid_cells(GRID)
    objs = {c["cell"]: 10.0 for c in cells}
    objs["c16"] = 1.0                                # thres 0.3, sigma 0.02
    r = C.rand_range(cells, objs, "c16", factor=1.5, min_halfwidth=0.3)
    assert r["thres_range"] == [0.21, 0.39] and r["sigma_range"] == [0.014, 0.026]
