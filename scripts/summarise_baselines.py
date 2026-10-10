"""Index the Phase 4 baseline results and check the set is complete.

    python scripts/summarise_baselines.py

Reads results/baselines/<arm>/*.json, cm_tune.json and eraft_reproduction.json; writes
results/baselines/summary.json (``kind: baselines_index``; every row carries its tier and tags)
and exits non-zero if a planned unit is missing, is a smoke run, or was produced from a dirty
tree. Planned units: {zero_flow, cm_baseline, R1} x (mvsec_test 4 sequences x dt 1, 4 +
dsec_test 3 sequences).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from check_data import sha256_file

from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results" / "baselines"
MVSEC = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
DSEC = ["thun_00_a", "zurich_city_02_a", "zurich_city_11_a"]


def planned() -> list[Path]:
    out = []
    for arm in ("zero_flow", "cm_baseline", "eraft"):
        for s in MVSEC:
            for dt in (1, 4):
                a = "eraft_mvsec20" if arm == "eraft" else arm
                out.append(BASE / a / f"mvsec_{s}_dt{dt}.json")
        for s in DSEC:
            a = "eraft_dsec" if arm == "eraft" else arm
            out.append(BASE / a / f"dsec_{s}_100ms.json")
    return out


def main() -> int:
    problems, rows = [], []
    for p in planned():
        rel = p.relative_to(REPO).as_posix()
        if not p.exists():
            problems.append(f"missing {rel}")
            continue
        d = json.loads(p.read_text())
        if d.get("smoke") or d.get("git_dirty"):
            problems.append(f"{rel}: smoke={d.get('smoke')} git_dirty={d.get('git_dirty')}")
        m = d["metrics"]
        rows.append({
            "file": rel, "sha256": sha256_file(p), "tier": d["tier"], "arm": d["arm"],
            "model": d["model"], "tags": d.get("tags", []), "dataset": d["dataset"],
            "sequence": d["sequence"], "dt": d.get("dt"), "window_ms": d.get("window_ms"),
            "epe": m["epe"], "epe_ci95": d["ci95"]["epe"], "outlier_pct": m["outlier_pct"],
            "npe_3": m["npe_3"], "fwl": m["fwl"], "n_windows": d["n_windows"],
            "n_windows_skipped_no_valid_px": d["n_windows_skipped_no_valid_px"],
            "git_sha": d["git_sha"],
        })
    extra = {}
    for name in ("cm_tune.json", "eraft_reproduction.json"):
        p = BASE / name
        if not p.exists():
            problems.append(f"missing results/baselines/{name}")
            continue
        d = json.loads(p.read_text())
        if d.get("smoke") or d.get("git_dirty"):
            problems.append(f"{name}: smoke={d.get('smoke')} git_dirty={d.get('git_dirty')}")
        extra[name] = {"sha256": sha256_file(p), "tier": d["tier"], "arm": d["arm"],
                       "git_sha": d["git_sha"]}
        if name == "eraft_reproduction.json":
            extra[name]["gate"] = d["gate"]
            if not d["gate"]["pass"]:
                problems.append("reproduction gate not met (see eraft_reproduction.json)")
        else:
            extra[name]["argmin"] = d["argmin"]["cell"]
    sha, dirty = git_sha(), git_dirty()
    doc = {
        "kind": "baselines_index",
        "phase": 4,
        "note": "T1 rows only (real events, real GT); R1 rows carry reference_external, and "
                "reference_seen_data on dsec_test.",
        "units": rows,
        "n_planned": len(planned()),
        "n_present": len(rows),
        "selection_and_gate": extra,
        "problems": problems,
        "git_sha": sha,
        "git_dirty": dirty,
        "created_by": "scripts/summarise_baselines.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (BASE / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(rows)}/{len(planned())} units; problems: {problems or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
