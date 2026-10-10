"""Choose the contrast-maximisation loss weights per self-supervised arm from the sweep runs
(Phase 5; configs/evflow.yaml ``losses.cm.sweep``; D-044) → ``results/train/cm_sweep.json``.

Each cell (smoothness_weight × divergence_penalty) is a short run on the arm's own training
events, selected on the arm's own tune stream (``mvsec_tune`` only; FWL, dt = 1). No GT enters
the choice. A cell is eligible only if it passes two label-free collapse checks on its tune
diagnostics (``losses.cm.cm_diagnostics``, D-043):
  (a) at its best epoch, the non-empty IWE pixel fraction is >= ``MIN_NONEMPTY`` of zero flow's;
  (b) the final-epoch tune flow magnitude is <= ``MAX_MAG_GROWTH`` x the median over its epochs.
Choice = argmax best-epoch tune FWL among eligible cells.

    python scripts/freeze_cm_sweep.py --sweep-root runs/train-sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "train" / "cm_sweep.json"
ARMS = ("real_selfsup", "v2e_selfsup")
MIN_NONEMPTY = 0.5
MAX_MAG_GROWTH = 2.0


def cell_suffix(s: float, d: float) -> str:
    return f"_sw{s:g}_dp{d:g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep-root", default=None, help="default: <paths.outputs.runs>/train-sweep")
    a = ap.parse_args(argv)
    paths = yaml.safe_load((REPO / "configs" / "paths.local.yaml").read_text())
    root = Path(a.sweep_root) if a.sweep_root else Path(paths["outputs"]["runs"]) / "train-sweep"
    sweep = yaml.safe_load((REPO / "configs" / "evflow.yaml").read_text())["losses"]["cm"]["sweep"]
    rep = yaml.safe_load((REPO / "configs" / "evflow.yaml").read_text())["train"]["representation"]
    doc = {"kind": "cm_weight_sweep", "phase": 5, "split": "mvsec_tune", "grid": sweep,
           "rule": {"metric": "own-stream tune dt1 FWL at the cell's best epoch, max",
                    "min_iwe_nonempty_ratio": MIN_NONEMPTY,
                    "max_final_flow_mag_over_median": MAX_MAG_GROWTH},
           "arms": {}}
    ok = True
    for arm in ARMS:
        cells = []
        for s in sweep["smoothness_weight"]:
            for d in sweep["divergence_penalty"]:
                rid = f"{arm}_{rep}_s{1337}{cell_suffix(s, d)}"
                rd = root / rid
                cell = {"run_id": rid, "smoothness_weight": s, "divergence_penalty": d}
                if not (rd / "DONE").exists():
                    cell["status"] = "missing"
                    cells.append(cell)
                    continue
                sel = json.loads((rd / "selection.json").read_text())
                stream = sel["stream"]
                with open(rd / "metrics.csv") as f:
                    rows = list(csv.DictReader(f))
                best = next(r for r in rows if int(r["epoch"]) == sel["selected_epoch"])
                mags = [float(r[f"{stream}_dt1_flow_mag"]) for r in rows]
                nonempty = float(best[f"{stream}_dt1_diag_iwe_nonempty_ratio"])
                growth = mags[-1] / max(statistics.median(mags), 1e-9)
                cell.update(status="done", stream=stream, selected_epoch=sel["selected_epoch"],
                            tune_fwl=sel["value"], iwe_nonempty_ratio=nonempty,
                            final_flow_mag_over_median=growth,
                            checkpoint_sha256=sel["checkpoint_sha256"],
                            eligible=nonempty >= MIN_NONEMPTY and growth <= MAX_MAG_GROWTH)
                cells.append(cell)
        elig = [c for c in cells if c.get("eligible")]
        chosen = max(elig, key=lambda c: c["tune_fwl"]) if elig else None
        ok &= chosen is not None
        doc["arms"][arm] = {"cells": cells, "chosen": None if chosen is None else {
            "smoothness_weight": chosen["smoothness_weight"],
            "divergence_penalty": chosen["divergence_penalty"], "run_id": chosen["run_id"],
            "tune_fwl": chosen["tune_fwl"]}}
        print(f"{arm}: " + (f"smoothness {chosen['smoothness_weight']}, divergence "
                            f"{chosen['divergence_penalty']} (tune FWL {chosen['tune_fwl']:.4f})"
                            if chosen else "NO eligible cell"))
    doc.update(git_sha=git_sha(), git_dirty=git_dirty(), created_by="scripts/freeze_cm_sweep.py",
               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
