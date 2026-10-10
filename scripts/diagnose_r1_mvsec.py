"""Diagnose R1 (E-RAFT 20 Hz MVSEC checkpoint) on mvsec_test: bug or generalisation gap?

    python scripts/diagnose_r1_mvsec.py

R1 is worse than zero flow on the indoor_flying sequences while it is far better on outdoor_day1
(results/baselines/{eraft_mvsec20,zero_flow}/). This script separates a pipeline error (sign,
axis swap, scale, warm start) from poor generalisation, and writes
results/baselines/r1_mvsec_diagnostic.json. Test-sequence statistics: reported only, never used
for any choice (CLAUDE.md rule 4).

Per sequence (dt = 1):
- from the committed units' per-window arrays: fraction of windows where R1's EPE < zero flow's,
  median EPE of both;
- on 200 evenly spaced windows, re-running R1 with and without warm start: least-squares scale
  of predicted on GT flow (sum p.g / sum g.g; a sign error gives < 0, a units error != ~1), median
  per-pixel cosine between predicted and GT flow, EPE with the same masks as the units, and the
  same three after swapping the predicted u/v channels (an axis swap would improve them).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from check_data import redact_home, sha256_file

from evflow.eval.datasets import load_yaml, mvsec_unit
from evflow.eval.run import ERAFTPredictor
from evflow.eval.windows import events_in_window, region_mask
from evflow.models.eraft_runner import WarmStartRunner, load_eraft
from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
NPZ = REPO / "runs" / "phase4-baselines" / "npz"
OUT = REPO / "results" / "baselines" / "r1_mvsec_diagnostic.json"
SEQS = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
N_SAMPLE = 200


def direction_stats(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    num = sum(float((p * g).sum()) for p, g in pairs)
    den = sum(float((g * g).sum()) for p, g in pairs)
    cos = np.concatenate([(p * g).sum(0) / (np.linalg.norm(p, axis=0) * np.linalg.norm(g, axis=0)
                                           + 1e-12) for p, g in pairs])
    epe = [float(np.linalg.norm(p - g, axis=0).mean()) for p, g in pairs]
    return {"ls_scale": num / den, "median_cosine": float(np.median(cos)),
            "epe": float(np.mean(epe))}


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    ev_cfg = load_yaml("configs/eval_mvsec.yaml")
    er = load_yaml("configs/eraft.yaml")
    ck = er["checkpoints"]["eraft_mvsec20"]
    model = load_eraft(paths["checkpoints"][ck["key"]], sha256=ck["sha256"],
                       num_bins=er["num_bins"], device="cuda")
    r = ev_cfg["regions"][er["mvsec_input_crop"]]
    crop = (r["rows"][0], r["rows"][1], r["cols"][0], r["cols"][1])
    rows = {}
    for seq in SEQS:
        a = np.load(NPZ / "eraft_mvsec20" / f"mvsec_{seq}_dt1.npz")
        z = np.load(NPZ / "zero_flow" / f"mvsec_{seq}_dt1.npz")
        ok = np.isfinite(a["epe"]) & np.isfinite(z["epe"])
        row = {"from_units": {
            "frac_windows_r1_better_than_zero": float(np.mean(a["epe"][ok] < z["epe"][ok])),
            "median_epe_r1": float(np.median(a["epe"][ok])),
            "median_epe_zero": float(np.median(z["epe"][ok])),
            "n_windows": int(ok.sum())}}
        reg = ev_cfg["regions"][ev_cfg["sequences"][seq]["region"]]
        windows, data, _ = mvsec_unit(paths, seq, 1, split="mvsec_test", region=reg)
        pick = np.unique(np.linspace(0, len(windows) - 1, N_SAMPLE).round().astype(int))
        region = region_mask(reg, data.height, data.width)
        for warm in (True, False):
            pred = ERAFTPredictor(WarmStartRunner(model, iters=er["iters"]), data.slicer,
                                  bins=er["num_bins"], height=data.height, width=data.width,
                                  device="cuda", crop=crop)
            pairs, swapped = [], []
            for k in pick:
                w = windows[k]
                if not warm:
                    pred._state.clear()
                ev = data.slicer.get_events(w.t_start_us, w.t_end_us)
                f = pred(w, ev)
                g, v = data.gt(w)
                m = v & region & events_in_window(ev["x"], ev["y"], data.height, data.width)
                if m.any():
                    pairs.append((f[:, m], g[:, m]))
                    swapped.append((f[::-1][:, m], g[:, m]))
            row["warm_start" if warm else "cold_start"] = {
                **direction_stats(pairs), "uv_swapped": direction_stats(swapped),
                "n_windows": len(pairs)}
        rows[seq] = row
        print(seq, json.dumps(row["from_units"]), json.dumps(row["warm_start"]), flush=True)

    doc = {
        "tier": "T1",
        "arm": "eraft_mvsec20",
        "model": "R1",
        "seed": None,
        "checkpoint_sha256": ck["sha256"],
        "dataset": "mvsec",
        "sequence": "mvsec_test (indoor_flying1-3, outdoor_day1)",
        "dt": 1,
        "representation": f"voxel{er['num_bins']} (E-RAFT convention)",
        "tags": ["reference_external", "diagnostic"],
        "purpose": "pipeline-error vs generalisation check of R1 on mvsec_test; reported only, "
                   "never used for a choice",
        "metrics": {s: rows[s]["warm_start"] for s in SEQS},
        "ci95": None,
        "n_windows": {s: rows[s]["warm_start"]["n_windows"] for s in SEQS},
        "per_sequence": rows,
        "sample": f"{N_SAMPLE} evenly spaced dt=1 windows per sequence; masks as the units",
        "inputs": {s: {"r1_npz_sha256": sha256_file(NPZ / "eraft_mvsec20" / f"mvsec_{s}_dt1.npz"),
                       "zero_npz_sha256": sha256_file(NPZ / "zero_flow" / f"mvsec_{s}_dt1.npz")}
                   for s in SEQS},
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "created_by": "scripts/diagnose_r1_mvsec.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"-> {redact_home(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
