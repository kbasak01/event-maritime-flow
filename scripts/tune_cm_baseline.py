"""Choose M0b's tile size and TV weight on mvsec_tune only (Phase 4 step 1; CLAUDE.md rule 4).

    python scripts/tune_cm_baseline.py [--max-windows N] [--only-cell K]

Grid and fixed parameters: configs/cm_baseline.yaml. Every cell is evaluated on the same
``tune.n_windows`` evenly spaced dt=1 windows of the tune slice (outdoor_day2), sparse mask, region
``center_256_no_hood``; the argmin of mean EPE (ties: higher FWL) is the cell every test unit uses
(scripts/run_baselines.py reads it from the JSON written here). Each cell's per-window arrays
are cached under runs/phase4-baselines/cm_tune/ so the job is resumable; ``--only-cell K``
computes and caches cell K only (the job runs cells in parallel, then one call without it writes
the JSON). Zero flow on the same windows is reported beside the grid as a reference.
``--max-windows`` is a smoke run that writes under runs/ only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from check_data import redact_home, sha256_file

from evflow.eval.datasets import load_yaml, mvsec_unit
from evflow.eval.run import CMPredictor, ZeroPredictor, evaluate_windows
from evflow.eval.summary import summarise
from evflow.models.cm_baseline import CMParams
from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "baselines" / "cm_tune.json"
CACHE = REPO / "runs" / "phase4-baselines" / "cm_tune"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--only-cell", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = load_yaml("configs/cm_baseline.yaml")
    tune = cfg["tune"]
    if tune["split"] != "mvsec_tune":
        raise SystemExit("M0b may only be tuned on mvsec_tune")
    ev_cfg = load_yaml("configs/eval_mvsec.yaml")
    boot = load_yaml("configs/evflow.yaml")["metrics"]["bootstrap"]
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    region = ev_cfg["regions"][tune["region"]]
    windows, data, meta = mvsec_unit(paths, tune["sequence"], int(tune["dt"]),
                                     split=tune["split"], region=region)
    n = int(tune["n_windows"]) if args.max_windows is None else int(args.max_windows)
    pick = np.unique(np.linspace(0, len(windows) - 1, n).round().astype(int))
    windows = [windows[i] for i in pick]
    cache = CACHE / ("smoke" if args.max_windows is not None else "full")
    cache.mkdir(parents=True, exist_ok=True)

    keys = sorted(cfg["grid"])
    grid = list(itertools.product(*(cfg["grid"][k] for k in keys)))
    if args.only_cell is not None and not 0 <= args.only_cell < len(grid):
        raise SystemExit(f"--only-cell must be in [0, {len(grid)})")
    cells = []
    for ci, values in enumerate(grid):
        if args.only_cell is not None and ci != args.only_cell:
            continue
        cell = dict(zip(keys, values, strict=True))
        params = CMParams.from_dict({**cfg["base"], **cell})
        name = "_".join(f"{k}{v}" for k, v in cell.items())
        key = hashlib.sha256(json.dumps({"params": params.to_dict(),
                                         "windows": [w.index for w in windows]},
                                        sort_keys=True).encode()).hexdigest()[:12]
        npz = cache / f"{name}_{key}.npz"   # keyed by parameters and window set
        t0 = time.time()
        if npz.exists():
            pw = dict(np.load(npz))
        else:
            pred = CMPredictor(params, data.height, data.width, args.device)
            pw = evaluate_windows(windows, data, pred,
                                  log=lambda s, name=name: print(f"[{name}] {s}", flush=True))
            np.savez_compressed(npz, **pw)
        s = summarise(pw, block_seconds=float(boot["block_seconds"]),
                      resamples=int(boot["resamples"]), ci=float(boot["ci"]), seed=0)
        cells.append({"cell": cell, "params": params.to_dict(), **s,
                      "per_window_npz_sha256": sha256_file(npz)})
        print(f"[{name}] EPE {s['metrics']['epe']:.4f} FWL {s['metrics']['fwl']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    if args.only_cell is not None:
        return 0
    zero = summarise(evaluate_windows(windows, data, ZeroPredictor(data.height, data.width)),
                     block_seconds=float(boot["block_seconds"]), resamples=int(boot["resamples"]),
                     ci=float(boot["ci"]), seed=0)
    best = min(cells, key=lambda c: (c["metrics"]["epe"], -c["metrics"]["fwl"]))
    sha, dirty = git_sha(), git_dirty()
    doc = {
        "tier": "T1",
        "arm": "cm_baseline_tune",
        "model": "M0b",
        "seed": 0,
        "checkpoint_sha256": None,
        "dataset": "mvsec",
        "sequence": tune["sequence"],
        "dt": int(tune["dt"]),
        "representation": "events (IWE)",
        "split": tune["split"],
        "purpose": "hyperparameter selection for M0b (tune split only, never a test split)",
        "metrics": best["metrics"],
        "ci95": best["ci95"],
        "n_windows": best["n_windows"],
        "window_aps_indices": [w.index for w in windows],
        "protocol": {**meta, "region_name": tune["region"], "region": region,
                     "mask": "valid_gt AND events_in_window (sparse) AND region",
                     "objective": tune["objective"], "tie_break": tune["tie_break"]},
        "config_sha256": sha256_file(REPO / "configs/cm_baseline.yaml"),
        "argmin": {"cell": best["cell"], "params": best["params"]},
        "grid": cells,
        "zero_flow_reference": {"metrics": zero["metrics"], "ci95": zero["ci95"],
                                "n_windows": zero["n_windows"]},
        "smoke": args.max_windows is not None,
        "git_sha": sha,
        "git_dirty": dirty,
        "created_by": "scripts/tune_cm_baseline.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = OUT if args.max_windows is None else cache / "cm_tune_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"argmin {best['cell']} EPE {best['metrics']['epe']:.4f} -> {redact_home(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
