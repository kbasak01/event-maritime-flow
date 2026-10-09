"""v2e contrast-threshold calibration on ``mvsec_tune`` only (IMPLEMENTATION_PLAN Phase 2 step 3).

Grid (``configs/v2e_profiles.yaml: calibration_grid``) over pos_thres = neg_thres, sigma_thres,
cutoff_hz and shot_noise_rate_hz. For each cell, v2e converts the calibration segments of the tune
slice (profile ``mvsec_grid`` + cell overrides, one runner unit per segment, same emulator seed in
every cell); §1.6 statistics of the synthetic events are compared with the real DVS events of the
same windows (``sim/stats.py``); objective = mean normalised W1. The argmin becomes ``mvsec_cal``;
the CT-randomisation range is the span of the thresholds (and sigmas) of all cells within
``rand_rule.factor`` × the minimum objective, widened to at least ±``rand_rule.min_halfwidth``
around the argmin. The count-image NCC threshold for the Phase 3 alignment check is
``ncc_rule.fraction`` × the tune median NCC of the argmin cell, with a frame-shift negative
control.

    python -m evflow.sim.calibrate cell --cell c07 --work DIR    # one grid cell (GPU)
    python -m evflow.sim.calibrate real --work DIR               # real-event statistics
    python -m evflow.sim.calibrate summarise --work DIR      # -> results/v2e_calibration.json

Guard: every segment and every window must lie inside ``splits/mvsec_tune.json``; any other
range raises before v2e or a store is touched.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from evflow.io.mvsec import read_aps_timestamps_us
from evflow.io.store import EventSlicer, EventStore
from evflow.sim import stats as S

REPO = Path(__file__).resolve().parents[3]
TUNE_SPLIT = REPO / "splits" / "mvsec_tune.json"
PROFILES = REPO / "configs" / "v2e_profiles.yaml"
PATHS = REPO / "configs" / "paths.local.yaml"
RESULT = REPO / "results" / "v2e_calibration.json"
GRID_FLAGS = {"pos_neg_thres": ("--pos_thres", "--neg_thres"), "sigma_thres": ("--sigma_thres",),
              "cutoff_hz": ("--cutoff_hz",), "shot_noise_rate_hz": ("--shot_noise_rate_hz",)}


class LeakageError(ValueError):
    """A calibration range outside mvsec_tune."""


# ---------------------------------------------------------------------------------------------
# pure rules (unit-tested)

def grid_cells(grid: dict) -> list[dict]:
    """Cells in a fixed order: thres (slowest) × sigma × cutoff × shot noise (fastest)."""
    keys = list(GRID_FLAGS)
    cells = []
    for i, vals in enumerate(itertools.product(*(grid[k] for k in keys))):
        cells.append({"cell": f"c{i:02d}", **{k: float(v) for k, v in zip(keys, vals,
                                                                            strict=True)}})
    return cells


def cell_overrides(cell: dict) -> dict[str, str]:
    out = {}
    for k, flags in GRID_FLAGS.items():
        for f in flags:
            out[f] = repr(float(cell[k]))
    return out


def tune_aps_range(tune: dict) -> tuple[int, int, int, int]:
    """(aps_first, aps_last, t_start_us, t_end_us) of the single tune range."""
    if tune.get("split") != "mvsec_tune" or len(tune["ranges_us"]) != 1:
        raise LeakageError("calibration needs splits/mvsec_tune.json with one range")
    a = tune["aps"][0]
    t0, t1 = tune["ranges_us"][0]
    return int(a["aps_first"]), int(a["aps_last"]), int(t0), int(t1)


def segments(tune: dict, n: int, n_frames: int) -> list[tuple[int, int]]:
    """``n`` segments of ``n_frames`` APS frames, starts evenly spaced over the tune range."""
    f0, f1, _, _ = tune_aps_range(tune)
    span = f1 - f0 + 1
    if n * n_frames > span:
        raise ValueError("segments do not fit in the tune slice")
    starts = np.linspace(f0, f1 - n_frames + 1, n).round().astype(int)
    segs = [(int(s), int(s) + n_frames - 1) for s in starts]
    check_inside_tune(tune, segs)
    return segs


def check_inside_tune(tune: dict, segs: list[tuple[int, int]]) -> None:
    f0, f1, _, _ = tune_aps_range(tune)
    for a, b in segs:
        if not (f0 <= a <= b <= f1):
            raise LeakageError(f"APS range [{a}, {b}] is outside mvsec_tune [{f0}, {f1}]")


def check_windows_inside_tune(tune: dict, wins: np.ndarray) -> None:
    _, _, t0, t1 = tune_aps_range(tune)
    if wins.size and (wins[:, 0].min() < t0 or wins[:, 1].max() > t1):
        raise LeakageError("a statistics window lies outside mvsec_tune")


def stat_windows(aps_us: np.ndarray, seg: tuple[int, int], warmup: int,
                 block_us: int) -> tuple[np.ndarray, np.ndarray]:
    """APS dt=1 windows of a segment after ``warmup`` frames, and fixed blocks over the same
    span (last partial block dropped)."""
    a, b = seg
    k = np.arange(a + warmup, b)
    wins = np.stack([aps_us[k], aps_us[k + 1]], 1)
    t0, t1 = int(wins[0, 0]), int(wins[-1, 1])
    starts = np.arange(t0, t1 - block_us + 1, block_us, dtype=np.int64)
    return wins, np.stack([starts, starts + block_us], 1)


def rand_range(cells: list[dict], objectives: dict[str, float], argmin: str, factor: float,
               min_halfwidth: float) -> dict:
    """CT-randomisation range rule (plan Phase 2 step 3)."""
    best = objectives[argmin]
    near = [c for c in cells if objectives[c["cell"]] <= factor * best]
    am = next(c for c in cells if c["cell"] == argmin)
    out = {"near_cells": [c["cell"] for c in near], "factor": factor,
           "min_halfwidth": min_halfwidth}
    for key, name in (("pos_neg_thres", "thres_range"), ("sigma_thres", "sigma_range")):
        lo = min(c[key] for c in near)
        hi = max(c[key] for c in near)
        out[f"{name}_near"] = [lo, hi]
        out[name] = [round(min(lo, am[key] * (1 - min_halfwidth)), 6),
                     round(max(hi, am[key] * (1 + min_halfwidth)), 6)]
    return out


# ---------------------------------------------------------------------------------------------
# I/O helpers

def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.suffix in (".yaml", ".yml") else json.loads(
        path.read_text())


def _ctx() -> dict:
    prof = _load(PROFILES)
    grid = prof["calibration_grid"]
    tune = json.loads(TUNE_SPLIT.read_text())
    paths = _load(PATHS)
    segs = segments(tune, int(grid["segments"]["n"]), int(grid["segments"]["n_frames"]))
    return {"prof": prof, "grid": grid, "tune": tune, "paths": paths, "segs": segs}


def _aps_us(paths: dict) -> np.ndarray:
    return read_aps_timestamps_us(Path(paths["mvsec"]["root"]) / "outdoor_day" /
                                  "outdoor_day2_data.hdf5")


def _store_events(path: Path) -> dict:
    with EventStore(path) as s:
        return {"t": s.events["t"][:] + s.t_offset, "x": s.events["x"][:],
                "y": s.events["y"][:], "p": s.events["p"][:]}


def _concat(evs: list[dict]) -> dict:
    return {k: np.concatenate([e[k] for e in evs]) for k in ("t", "x", "y", "p")}


def _segment_stats(events_per_seg: list[dict], aps: np.ndarray, ctx: dict) -> dict:
    """Stats pooled over segments: histograms add up; windows per segment."""
    g = ctx["grid"]
    hist = {k: np.zeros(len(S.EDGES[k]) - 1, np.int64) for k in S.STATS}
    n_ev = 0
    for e, seg in zip(events_per_seg, ctx["segs"], strict=True):
        wins, blocks = stat_windows(aps, seg, int(g["warmup_frames"]), int(g["block_ms"]) * 1000)
        check_windows_inside_tune(ctx["tune"], wins)
        st = S.slice_stats(e, wins, blocks, 346, 260)
        for k in S.STATS:
            hist[k] += st["hist"][k]
        n_ev += st["n_events"]
    return {"hist": {k: v.tolist() for k, v in hist.items()}, "n_events": n_ev}


def _real_segment_events(ctx: dict, aps: np.ndarray) -> list[dict]:
    store = Path(ctx["paths"]["outputs"]["events"]) / "mvsec_outdoor_day2_left.h5"
    out = []
    with EventSlicer(store) as sl:
        for a, b in ctx["segs"]:
            e = sl.get_events(int(aps[a]), int(aps[b]))
            if e is None:
                raise RuntimeError(f"real store does not cover APS {a}..{b}")
            out.append(e)
    return out


def _unit_name(cell: str, j: int) -> str:
    return f"{cell}_seg{j}"


# ---------------------------------------------------------------------------------------------
# commands

def cmd_cell(cell_id: str, work: Path, seed: int) -> dict:
    ctx = _ctx()
    cells = {c["cell"]: c for c in grid_cells(ctx["grid"])}
    cell = cells[cell_id]
    out_json = work / "stats" / f"{cell_id}.json"
    if out_json.exists():
        return json.loads(out_json.read_text())
    aps = _aps_us(ctx["paths"])
    evs, units = [], []
    for j, (a, b) in enumerate(ctx["segs"]):
        unit = _unit_name(cell_id, j)
        argv = ["mvsec", "--sequence", "outdoor_day2", "--aps", str(a), str(b),
                "--frames-cache", str(work / "aps_cache"), "--profile",
                ctx["grid"]["profile"], "--unit", unit, "--out", str(work / "units"),
                "--seed", str(seed)]
        argv += [f"--set={k}={v}" for k, v in cell_overrides(cell).items()]
        subprocess.run([sys.executable, "-m", "evflow.sim.v2e_runner", *argv], check=True)
        man = json.loads((work / "units" / f"{unit}.json").read_text())
        units.append({"unit": unit, "aps": [a, b], "store_sha256": man["store_sha256"],
                      "run_manifest_sha256": man["run_manifest_sha256"],
                      "wall_time_s": man["wall_time_s"],
                      "undersampling_warnings": man["warnings"]["undersampling"]})
        evs.append(_store_events(work / "units" / f"{unit}.h5"))
    doc = {**cell, "overrides": cell_overrides(cell), "units": units,
           **_segment_stats(evs, aps, ctx)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc) + "\n")
    return doc


def cmd_real(work: Path) -> dict:
    ctx = _ctx()
    out_json = work / "stats" / "real.json"
    aps = _aps_us(ctx["paths"])
    # export the segments' APS frames once, before cells run concurrently on the same cache
    from evflow.sim.v2e_runner import export_mvsec_aps

    src = Path(ctx["paths"]["mvsec"]["root"]) / "outdoor_day" / "outdoor_day2_data.hdf5"
    for a, b in ctx["segs"]:
        export_mvsec_aps(src, work / "aps_cache" / f"outdoor_day2_left_aps_{a:06d}_{b:06d}", a, b)
    doc = _segment_stats(_real_segment_events(ctx, aps), aps, ctx)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc) + "\n")
    return doc


def _ncc(ctx: dict, aps: np.ndarray, work: Path, cell_id: str) -> dict:
    g = ctx["grid"]
    rule = g["ncc_rule"]
    real = _real_segment_events(ctx, aps)
    per, shifted, mirrored = [], [], []
    for j, seg in enumerate(ctx["segs"]):
        syn = _store_events(work / "units" / f"{_unit_name(cell_id, j)}.h5")
        wins, _ = stat_windows(aps, seg, int(g["warmup_frames"]), int(g["block_ms"]) * 1000)
        check_windows_inside_tune(ctx["tune"], wins)
        per.append(S.count_ncc(real[j], syn, wins, 346, 260, blur_sigma=rule["blur_sigma_px"]))
        k = int(rule["control_shift_frames"])
        # negative control: real window i vs synthetic window i + k
        syn_shift = {**syn, "t": syn["t"] - (aps[seg[0] + k] - aps[seg[0]])}
        shifted.append(S.count_ncc(real[j], syn_shift, wins[: len(wins) - k], 346, 260,
                                   blur_sigma=rule["blur_sigma_px"]))
        # negative control: geometry error (x mirrored)
        syn_mirror = {**syn, "x": (345 - syn["x"].astype(np.int64)).astype(np.uint16)}
        mirrored.append(S.count_ncc(real[j], syn_mirror, wins, 346, 260,
                                    blur_sigma=rule["blur_sigma_px"]))
    v, vs, vm = np.concatenate(per), np.concatenate(shifted), np.concatenate(mirrored)
    med = float(np.nanmedian(v))
    thr = float(rule["fraction"]) * med
    return {"cell": cell_id, "blur_sigma_px": rule["blur_sigma_px"], "n_windows": int(
        np.isfinite(v).sum()), "median": med, "p10": float(np.nanpercentile(v, 10)),
        "p90": float(np.nanpercentile(v, 90)), "threshold": thr, "rule": rule,
        "control_shift_frames": int(rule["control_shift_frames"]),
        "control_median": float(np.nanmedian(vs)),
        "control_below_threshold": bool(np.nanmedian(vs) < thr),
        "mirror_control_median": float(np.nanmedian(vm)),
        "mirror_control_below_threshold": bool(np.nanmedian(vm) < thr)}


def cmd_summarise(work: Path, out: Path = RESULT) -> dict:
    ctx = _ctx()
    g = ctx["grid"]
    cells = grid_cells(g)
    real = json.loads((work / "stats" / "real.json").read_text())
    rh = {k: np.asarray(v) for k, v in real["hist"].items()}
    rows, objs = [], {}
    for c in cells:
        d = json.loads((work / "stats" / f"{c['cell']}.json").read_text())
        sh = {k: np.asarray(v) for k, v in d["hist"].items()}
        obj, per = S.objective(rh, sh)
        objs[c["cell"]] = obj
        rows.append({**c, "objective": obj, "w1_normalised": per, "n_events": d["n_events"],
                     "units": d["units"], "hist": d["hist"]})
    argmin = min(objs, key=lambda k: (objs[k], k))
    sens = sensitivity(rh, rows)
    rr = rand_range(cells, objs, argmin, float(g["rand_rule"]["factor"]),
                    float(g["rand_rule"]["min_halfwidth"]))
    aps = _aps_us(ctx["paths"])
    tune = ctx["tune"]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    doc = {
        "kind": "v2e_calibration", "tier": "T2",
        "tier_note": "event statistics of v2e(MVSEC APS) vs real DVS of the same sensor; "
                     "simulator-fidelity statistics, not flow accuracy",
        "plan_step": "Phase 2 step 3", "split": "mvsec_tune",
        "split_manifest_sha256": tune["manifest_sha256"],
        "segments_aps": ctx["segs"], "grid": {k: g[k] for k in GRID_FLAGS},
        "profile": g["profile"], "seed": int(g["seed"]),
        "stat_definition": {"module": "src/evflow/sim/stats.py", "stats": list(S.STATS),
                            "edges": {k: [float(S.EDGES[k][0]), float(S.EDGES[k][-1]),
                                          len(S.EDGES[k]) - 1] for k in S.STATS},
                            "warmup_frames": g["warmup_frames"], "block_ms": g["block_ms"],
                            "objective": "mean over stats of W1(real, syn) / std(real)"},
        "real": {"n_events": real["n_events"], "hist": real["hist"]},
        "cells": rows, "objectives": objs, "argmin": argmin,
        "argmin_params": next(c for c in cells if c["cell"] == argmin),
        "rand": rr, "ncc": _ncc(ctx, aps, work, argmin),
        "sensitivity": {"note": "reported only; the frozen choice is `argmin` (D-027)", **sens},
        "git_sha": sha, "created_by": "src/evflow/sim/calibrate.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def sensitivity(rh: dict, rows: list[dict]) -> dict:
    """Argmin under alternative objectives (reported, never used for the choice): the plan-literal
    unnormalised mean W1, and the normalised objective leaving each stat out."""
    def arg(f):
        vals = {r["cell"]: f(r) for r in rows}
        return min(vals, key=lambda k: (vals[k], k))

    hists = {r["cell"]: {k: np.asarray(v) for k, v in r["hist"].items()} for r in rows}
    raw = arg(lambda r: float(np.mean([S.w1_hist(rh[k], hists[r["cell"]][k], S.EDGES[k])
                                       for k in S.STATS])))
    loo = {k: arg(lambda r, k=k: float(np.mean([v for s, v in r["w1_normalised"].items()
                                                 if s != k])))
           for k in S.STATS}
    return {"argmin_unnormalised_mean_w1": raw, "argmin_leave_one_stat_out": loo}


def recompute_argmin(doc: dict) -> str:
    """Argmin re-derived from the stored histograms (auditor check)."""
    rh = {k: np.asarray(v) for k, v in doc["real"]["hist"].items()}
    objs = {c["cell"]: S.objective(rh, {k: np.asarray(v) for k, v in c["hist"].items()})[0]
            for c in doc["cells"]}
    return min(objs, key=lambda k: (objs[k], k))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--cell", required=True)
    c.add_argument("--work", type=Path, required=True)
    r = sub.add_parser("real")
    r.add_argument("--work", type=Path, required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--out", type=Path, default=RESULT)
    sub.add_parser("cells")
    a = ap.parse_args(argv)
    if a.cmd == "cells":
        for cell in grid_cells(_load(PROFILES)["calibration_grid"]):
            print(cell["cell"])
        return 0
    if a.cmd == "cell":
        seed = int(_load(PROFILES)["calibration_grid"]["seed"])
        d = cmd_cell(a.cell, a.work, seed)
        print(json.dumps({"cell": a.cell, "n_events": d["n_events"]}))
    elif a.cmd == "real":
        d = cmd_real(a.work)
        print(json.dumps({"real_n_events": d["n_events"]}))
    else:
        d = cmd_summarise(a.work, a.out)
        print(json.dumps({"argmin": d["argmin"], "objective": d["objectives"][d["argmin"]],
                          "rand": {k: d["rand"][k] for k in ("thres_range", "sigma_range")},
                          "ncc": {k: d["ncc"][k] for k in ("median", "threshold",
                                                           "control_median")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
