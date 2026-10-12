"""Phase 5 summary of every training run → ``results/train/summary.json``.

Reads ``<runs>/train/<run_id>/{manifest.json, selection.json, metrics.csv}`` for the expected
runs (``--expect`` run ids; default: every run directory), checks the checkpoint SHA-256 against
``selection.json``, and records per run: arm, representation, seed, selection rule / epoch /
value, checkpoint SHA, git provenance, and the tune metrics at the selected epoch per stream with
an explicit tier (real tune events + real GT = T1; v2e(tune) events + real GT = T2). Tune
metrics are selection evidence on ``mvsec_tune``, not test results. Runs listed with
``--exclude RUN_ID=REASON`` are kept in the file with the reason and ``excluded: true``.

    python scripts/summarise_training.py --expect real_sup_voxel5_s1337 ... \
        [--exclude v2e_selfsup_voxel5_s1337="collapse (gate finding 3)"]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from check_data import sha256_file

from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "train" / "summary.json"
TIER = {"real": "T1", "v2e": "T2"}
KEYS = ("epe", "outlier_pct", "npe_1", "npe_3", "ae_deg", "fwl", "flow_mag")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_record(run_dir: Path) -> dict:
    rec = {"run_id": run_dir.name, "status": "missing"}
    if not (run_dir / "manifest.json").exists():
        return rec
    man = json.loads((run_dir / "manifest.json").read_text())
    rec.update(arm=man["arm"], model=man["model"], representation=man["representation"],
               seed=man["seed"], loss=man["loss"], git_sha=man["git_sha"],
               git_dirty=man["git_dirty"], config_files=man["config_files"],
               manifest_sha256=sha256_file(run_dir / "manifest.json"),
               train_windows=man["train_windows"], n_params=man["n_params"],
               epochs_planned=man["epochs"])
    rows = []
    if (run_dir / "metrics.csv").exists():
        with open(run_dir / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
    rec["epochs_done"] = len(rows)
    if not (run_dir / "DONE").exists():
        rec["status"] = "incomplete"
        return rec
    sel = json.loads((run_dir / "selection.json").read_text())
    sha_ok = sha256_file(run_dir / sel["checkpoint"]) == sel["checkpoint_sha256"]
    rec.update(status="done" if sha_ok else "checkpoint_sha_mismatch",
               checkpoint_sha256=sel["checkpoint_sha256"],
               selection={k: sel[k] for k in ("rule", "split", "stream", "metric", "dt", "mode",
                                              "selected_epoch", "value")})
    row = next(r for r in rows if int(r["epoch"]) == sel["selected_epoch"])
    tune = {}
    for stream in man["tune_streams"]:
        for dt in man["config_resolved"]["train"]["dts"]:
            m = {k: _num(row.get(f"{stream}_dt{dt}_{k}")) for k in KEYS}
            tune[f"{stream}_dt{dt}"] = {
                "tier": TIER[stream], "events": stream, "dataset": "mvsec",
                "sequence": "outdoor_day2", "split": "mvsec_tune", "dt": dt,
                "used_for_selection": stream == sel["stream"] and dt == sel["dt"],
                "metrics": m}
    rec["tune_at_selected_epoch"] = tune
    diag = {k.split("_diag_", 1)[1]: _num(v) for k, v in row.items() if "_diag_" in k}
    rec["collapse_diagnostics_at_selected_epoch"] = diag
    rec["final_epoch_train_loss"] = _num(rows[-1].get("train_loss"))
    clipped = [_num(r.get("clipped_step_fraction")) for r in rows]
    if any(c is not None for c in clipped):        # logged from Phase 7 on (D-049, D-056)
        rec["clipped_step_fraction_max"] = max(c for c in clipped if c is not None)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=None, help="default: <paths.outputs.runs>/train")
    ap.add_argument("--expect", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="RUN_ID=REASON")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--phase", type=int, default=5)
    a = ap.parse_args(argv)
    paths = yaml.safe_load((REPO / "configs" / "paths.local.yaml").read_text())
    runs = Path(a.runs) if a.runs else Path(paths["outputs"]["runs"]) / "train"
    expect = a.expect if a.expect is not None else sorted(p.name for p in runs.iterdir()
                                                           if p.is_dir())
    excluded = dict(e.split("=", 1) for e in a.exclude)
    records = []
    for rid in expect:
        rec = run_record(runs / rid)
        if rid in excluded:
            rec.update(excluded=True, exclusion_reason=excluded[rid])
        records.append(rec)
    doc = {"kind": "training_runs", "phase": a.phase,
           "note": "Tune metrics are mvsec_tune selection evidence (T1 real events, T2 v2e "
                   "events, real GT); test results are in results/eval/.",
           "n_expected": len(expect),
           "n_done": sum(r["status"] == "done" for r in records),
           "n_excluded": sum(bool(r.get("excluded")) for r in records),
           "runs": records, "git_sha": git_sha(), "git_dirty": git_dirty(),
           "created_by": "scripts/summarise_training.py",
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{doc['n_done']}/{doc['n_expected']} runs done, {doc['n_excluded']} excluded -> {out}")
    return 0 if doc["n_done"] == doc["n_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
