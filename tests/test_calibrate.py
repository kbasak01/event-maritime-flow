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


def test_timing_grid_inherits_and_prefixes():
    prof = {"calibration_grid": {**GRID, "seed": 1337, "segments": {"n": 3, "n_frames": 175}},
            "calibration_grid_timing": {"base": "calibration_grid", "cell_prefix": "t",
                                        "pos_neg_thres": [0.4, 0.5], "cutoff_hz": [0]}}
    g = C.resolve_grid(prof, "calibration_grid_timing")
    assert g["seed"] == 1337 and g["segments"]["n"] == 3 and "base" not in g
    cells = C.grid_cells(g)
    assert [c["cell"] for c in cells][:2] == ["t00", "t01"] and len(cells) == 2 * 2 * 1 * 2
    assert all(c["cutoff_hz"] == 0.0 for c in cells)


def _row(cell, thres, sigma, cutoff, obj, peak):
    return {"cell": cell, "pos_neg_thres": thres, "sigma_thres": sigma, "cutoff_hz": cutoff,
            "shot_noise_rate_hz": 0.0, "objective": obj, "lag": {"peak_lag_frames": peak}}


def test_timing_choice_only_lag_eligible_cells():
    rows = [_row("c30", 0.4, 0.04, 60.0, 0.37, 0.5),        # best objective but lagged
            _row("t02", 0.4, 0.04, 0.0, 1.05, 0.0),
            _row("t06", 0.5, 0.04, 0.0, 0.60, 0.125),
            _row("t10", 0.6, 0.04, 0.0, 0.70, -0.125),
            _row("t14", 0.7, 0.02, 0.0, 2.00, 0.0)]
    ch = C.timing_choice(rows, 0.125, 1.5, 0.3)
    assert ch["argmin"] == "t06" and "c30" not in ch["eligible"]
    assert ch["rand"]["near_cells"] == ["t06", "t10"]          # within 1.5 x 0.60
    assert ch["rand"]["thres_range"] == [0.35, 0.65]            # +-30 % around 0.5 beats 0.5-0.6
    with pytest.raises(RuntimeError):
        C.timing_choice([rows[0]], 0.125, 1.5, 0.3)


def test_recompute_timing_argmin_from_hist():
    edges = {k: C.S.EDGES[k] for k in C.S.STATS}
    real = {k: np.bincount([5, 6, 7], minlength=len(edges[k]) - 1).tolist() for k in C.S.STATS}
    near = {k: np.bincount([5, 6, 8], minlength=len(edges[k]) - 1).tolist() for k in C.S.STATS}
    far = {k: np.bincount([5, 30, 60], minlength=len(edges[k]) - 1).tolist() for k in C.S.STATS}
    doc = {"real": {"hist": real}, "lag_rule": {"max_abs_peak_lag_frames": 0.125},
           "cells": [{"cell": "c30", "hist": real, "lag": {"peak_lag_frames": 0.5}},
                     {"cell": "t01", "hist": far, "lag": {"peak_lag_frames": 0.0}},
                     {"cell": "t02", "hist": near, "lag": {"peak_lag_frames": 0.0}}]}
    assert C.recompute_timing_argmin(doc) == "t02"
