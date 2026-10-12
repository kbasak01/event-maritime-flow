"""The Phase 7 evaluation matrix (IMPLEMENTATION_PLAN Phase 7 step 4; D-057): which models are
evaluated on which T3 units, where their JSON lives. Shared by the eval job, ``scripts/
summarise_maritime.py`` and ``scripts/make_tables.py``. Paths are repo-relative.

| tier     | units                                             | models                         |
|----------|---------------------------------------------------|--------------------------------|
| T3a, T3b | smd_test (3 videos), lars_eval_kf (class-sliced), | M0a, M0b, R1 (DSEC ckpt), the  |
|          | lars_eval_all                                     | 11 voxel5 MVSEC runs, 2 mar.   |
| T1       | real mvsec_test, dt 1 and 4                       | the 2 maritime runs            |
"""

from __future__ import annotations

import json
from pathlib import Path

from evflow.eval import study

REPO = Path(__file__).resolve().parents[3]

MARITIME_ARMS = ("maritime_selfsup", "maritime_teacher")
MARITIME_SEED = 1337
BASELINES = {"zero_flow": ("zero", "M0a"), "cm_baseline": ("cm", "M0b"),
             "eraft_dsec": ("eraft", "R1")}
TIERS = ("T3a", "T3b")
LARS_SEQS = ("lars_eval_kf", "lars_eval_all")
CLASS_SLICES = ("water", "sky", "obstacle")


def smd_videos(repo: Path = REPO) -> list[str]:
    return list(json.loads((repo / "splits/smd_test.json").read_text())["videos"])


def t3_units(repo: Path = REPO) -> list[tuple[str, str]]:
    return [("smd", v) for v in smd_videos(repo)] + [("lars", s) for s in LARS_SEQS]


def seed_dir(seed) -> str:
    return "none" if seed is None else str(seed)


def t3_path(arm: str, seed, dataset: str, sequence: str, tier: str) -> str:
    return f"results/eval_maritime/{arm}/{seed_dir(seed)}/{dataset}_{sequence}_{tier}.json"


def maritime_runs(repo: Path = REPO) -> list[dict]:
    p = repo / "results/train/maritime_summary.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    return [r for r in doc["runs"] if r.get("status") == "done" and not r.get("excluded")]


def models(repo: Path = REPO) -> list[dict]:
    """Every model of the T3 matrix: baselines, the voxel5 MVSEC runs, the maritime runs."""
    out = [{"arm": a, "seed": (0 if a == "cm_baseline" else None), "kind": k, "model": m,
            "run_id": None} for a, (k, m) in BASELINES.items()]
    for r in study.trained_runs(repo) + maritime_runs(repo):
        if r["representation"] == study.MAIN_REP:
            out.append({"arm": r["arm"], "seed": int(r["seed"]), "kind": "m1", "model": "M1",
                        "run_id": r["run_id"]})
    return out


def expected_units(repo: Path = REPO) -> list[dict]:
    out = []
    for m in models(repo):
        for ds, seq in t3_units(repo):
            for tier in TIERS:
                out.append({**m, "tier": tier, "dataset": ds, "sequence": seq,
                            "path": t3_path(m["arm"], m["seed"], ds, seq, tier)})
    for r in maritime_runs(repo):
        for s in study.MVSEC_SEQS:
            for dt in study.DTS:
                out.append({"arm": r["arm"], "seed": int(r["seed"]), "kind": "m1",
                            "model": "M1", "run_id": r["run_id"], "tier": "T1",
                            "dataset": "mvsec", "sequence": s, "dt": dt,
                            "path": study.m1_path(r["arm"], int(r["seed"]),
                                                  study.stem("mvsec", s, dt))})
    return out


def commands(repo: Path = REPO, runs_root: str = "runs/train") -> list[str]:
    """One shell command per (model, unit) of the matrix (both T3 tiers come from one call),
    then the T1 units of the maritime runs; the scripts skip units whose JSON exists."""
    out = []
    for m in models(repo):
        for ds, seq in t3_units(repo):
            a = (f"--model m1 --run-dir {runs_root}/{m['run_id']}" if m["kind"] == "m1"
                 else f"--model {m['kind']}")
            out.append(f"scripts/run_eval_maritime.py {a} --dataset {ds} --sequence {seq}")
    for r in maritime_runs(repo):
        for s in study.MVSEC_SEQS:
            for dt in study.DTS:
                out.append(f"scripts/run_eval.py --run-dir {runs_root}/{r['run_id']} "
                           f"--dataset mvsec --sequence {s} --dt {dt}")
    return out


if __name__ == "__main__":
    import sys
    for c in commands(runs_root=sys.argv[1] if len(sys.argv) > 1 else "runs/train"):
        print(c)
