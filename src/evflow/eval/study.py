"""The Phase 6 evaluation matrix (IMPLEMENTATION_PLAN Phase 6, D-051): which models are evaluated
on which units, where their summary JSON lives, and which cells are omitted by design.

Shared by ``scripts/summarise_eval.py`` (inventory, seed aggregates, paired statistics) and
``scripts/make_tables.py`` (rendering), so both read one definition. Paths are repo-relative.

| tier | events                         | M1 runs                              | baselines    |
|------|--------------------------------|--------------------------------------|--------------|
| T1   | real `mvsec_test`, dt 1 and 4  | all trained runs (voxel5 + ablation) | M0a, M0b, R1 |
| T1   | real `dsec_test`, 100 ms       | voxel5 runs                          | M0a, M0b, R1 |
| T2   | v2e(APS) `mvsec_test` (D-035)  | voxel5 runs                          | M0a          |
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

MVSEC_SEQS = ("indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1")
DTS = (1, 4)
DSEC_SEQS = ("thun_00_a", "zurich_city_02_a", "zurich_city_11_a")

CORE_ARMS = ("real_sup", "v2e_fixed_sup", "v2e_rand_sup")      # 3 seeds each (plan §1.4)
CORE_SEEDS = (1337, 2026, 7)                                   # configs/evflow.yaml seeds_core
SELFSUP_ARMS = ("real_selfsup", "v2e_selfsup")                 # 1 seed
M1_ARMS = CORE_ARMS + SELFSUP_ARMS
MAIN_REP = "voxel5"
ABLATION_REPS = ("voxel5", "voxel15", "timesurface", "count")  # plan §1.5, D-043(f)
ABLATION_ARMS = ("real_sup", "v2e_fixed_sup")
ABLATION_SEED = 1337

BASELINE_MODEL = {"zero_flow": "M0a", "cm_baseline": "M0b", "eraft_mvsec20": "R1",
                  "eraft_dsec": "R1"}

BY_DESIGN = ("not evaluated by design (D-051): T2 covers voxel5 M1 runs and M0a only; DSEC "
             "covers voxel5 M1 runs and the Phase 4 baselines")


def expected_seeds(arm: str, rep: str) -> tuple[int, ...]:
    return CORE_SEEDS if arm in CORE_ARMS and rep == MAIN_REP else (ABLATION_SEED,)


def stem(dataset: str, sequence: str, dt: int | None = None, *, rep: str = MAIN_REP,
         events: str = "real") -> str:
    tag = f"dt{dt}" if dataset == "mvsec" else "100ms"
    return (f"{dataset}_{sequence}_{tag}" + ("" if rep == MAIN_REP else f"_{rep}")
            + ("_v2e" if events == "v2e" else ""))


def m1_path(arm: str, seed: int, st: str) -> str:
    return f"results/eval/{arm}/{seed}/{st}.json"


def baseline_path(arm: str, st: str) -> str:
    return f"results/baselines/{arm}/{st}.json"


def trained_runs(repo: Path = REPO) -> list[dict]:
    """Done, non-excluded runs of ``results/train/summary.json`` (Phase 5)."""
    doc = json.loads((repo / "results/train/summary.json").read_text())
    return [r for r in doc["runs"] if r.get("status") == "done" and not r.get("excluded")]


def expected_units(runs: list[dict]) -> list[dict]:
    """Every evaluation unit Phase 6 must produce, with its tier and JSON path."""
    out = []
    mvsec = [(s, dt) for s in MVSEC_SEQS for dt in DTS]
    for r in runs:
        arm, seed, rep = r["arm"], int(r["seed"]), r["representation"]
        units = [("T1", "mvsec", s, dt, "real") for s, dt in mvsec]
        if rep == MAIN_REP:
            units += [("T1", "dsec", s, None, "real") for s in DSEC_SEQS]
            units += [("T2", "mvsec", s, dt, "v2e") for s, dt in mvsec]
        for tier, ds, s, dt, ev in units:
            out.append({"kind": "m1", "tier": tier, "arm": arm, "seed": seed,
                        "representation": rep, "dataset": ds, "sequence": s, "dt": dt,
                        "events": ev, "path": m1_path(arm, seed, stem(ds, s, dt, rep=rep,
                                                                          events=ev))})
    for arm in ("zero_flow", "cm_baseline", "eraft_mvsec20"):
        for s, dt in mvsec:
            out.append({"kind": "baseline", "tier": "T1", "arm": arm, "dataset": "mvsec",
                        "sequence": s, "dt": dt, "events": "real",
                        "path": baseline_path(arm, stem("mvsec", s, dt))})
    for arm in ("zero_flow", "cm_baseline", "eraft_dsec"):
        for s in DSEC_SEQS:
            out.append({"kind": "baseline", "tier": "T1", "arm": arm, "dataset": "dsec",
                        "sequence": s, "dt": None, "events": "real",
                        "path": baseline_path(arm, stem("dsec", s))})
    for s, dt in mvsec:
        out.append({"kind": "baseline", "tier": "T2", "arm": "zero_flow", "dataset": "mvsec",
                    "sequence": s, "dt": dt, "events": "v2e",
                    "path": baseline_path("zero_flow", stem("mvsec", s, dt, events="v2e"))})
    return out
