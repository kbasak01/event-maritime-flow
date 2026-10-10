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
- from the dt = 1 and dt = 4 units: the same, plus R1/zero mean-EPE ratio per motion band (bands of
  the window's zero-flow EPE = mean |GT| over the mask), so near-static windows cannot hide a gap;
- on 200 evenly spaced dt = 1 windows, re-running R1 warm (each pick preceded by its WARMUP
  consecutive predecessors, run as a chain and not scored) and cold: least-squares scale
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
WARMUP = 10          # predecessors run before each warm-start pick
BANDS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]   # zero-flow EPE bands (px)


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
        row = {"from_units": {}}
        for dt in (1, 4):
            a = np.load(NPZ / "eraft_mvsec20" / f"mvsec_{seq}_dt{dt}.npz")
            z = np.load(NPZ / "zero_flow" / f"mvsec_{seq}_dt{dt}.npz")
            ok = np.isfinite(a["epe"]) & np.isfinite(z["epe"])
            ea, ez = a["epe"][ok], z["epe"][ok]
            bands = []
            for lo, hi in zip(BANDS[:-1], BANDS[1:], strict=True):
                m = (ez >= lo) & (ez < hi)
                if m.sum() >= 20:
                    bands.append({"zero_epe_band_px": [lo, hi], "n_windows": int(m.sum()),
                                  "r1_over_zero_mean_epe": float(ea[m].mean() / ez[m].mean())})
            row["from_units"][f"dt{dt}"] = {
                "frac_windows_r1_better_than_zero": float(np.mean(ea < ez)),
                "mean_epe_r1": float(ea.mean()), "mean_epe_zero": float(ez.mean()),
                "median_epe_r1": float(np.median(ea)), "median_epe_zero": float(np.median(ez)),
                "n_windows": int(ok.sum()), "by_motion_band": bands}
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
                pred._state.clear()
                if warm:  # run the chain of predecessors so the pick is really warm-started
                    for j in range(max(0, k - WARMUP), k):
                        wj = windows[j]
                        pred(wj, data.slicer.get_events(wj.t_start_us, wj.t_end_us))
                    if k > 0 and windows[k - 1].t_end_us != w.t_start_us:
                        raise RuntimeError("dt=1 windows must be contiguous for warm start")
                ev = data.slicer.get_events(w.t_start_us, w.t_end_us)
                f = pred(w, ev)
                g, v = data.gt(w)
                m = v & region & events_in_window(ev["x"], ev["y"], data.height, data.width)
                if m.any():
                    pairs.append((f[:, m], g[:, m]))
                    swapped.append((f[::-1][:, m], g[:, m]))
            row["warm_start" if warm else "cold_start"] = {
                **direction_stats(pairs), "uv_swapped": direction_stats(swapped),
                "n_windows": len(pairs), **({"warmup_predecessors": WARMUP} if warm else {})}
        rows[seq] = row
        print(seq, json.dumps({k: v["mean_epe_r1"] for k, v in row["from_units"].items()}),
              "warm", json.dumps({k: row["warm_start"][k] for k in ("epe", "ls_scale")}),
              "cold", json.dumps({k: row["cold_start"][k] for k in ("epe", "ls_scale")}),
              flush=True)

    doc = {
        "tier": "T1",
        "arm": "eraft_mvsec20",
        "model": "R1",
        "seed": None,
        "checkpoint_sha256": ck["sha256"],
        "dataset": "mvsec",
        "sequence": "mvsec_test (indoor_flying1-3, outdoor_day1)",
        "dt": "1 (re-runs); 1 and 4 (from_units)",
        "representation": f"voxel{er['num_bins']} (E-RAFT convention)",
        "tags": ["reference_external", "diagnostic"],
        "purpose": "pipeline-error vs generalisation check of R1 on mvsec_test; reported only, "
                   "never used for a choice",
        "metrics": {s: {"warm_start": rows[s]["warm_start"], "cold_start": rows[s]["cold_start"]}
                    for s in SEQS},
        "ci95": None,
        "n_windows": {s: rows[s]["warm_start"]["n_windows"] for s in SEQS},
        "per_sequence": rows,
        "sample": f"{N_SAMPLE} evenly spaced dt=1 windows per sequence; masks as the units; warm "
                  f"arm: {WARMUP} unscored predecessors per pick",
        "inputs": {f"{s}_dt{dt}": {
            "r1_npz_sha256": sha256_file(NPZ / "eraft_mvsec20" / f"mvsec_{s}_dt{dt}.npz"),
            "zero_npz_sha256": sha256_file(NPZ / "zero_flow" / f"mvsec_{s}_dt{dt}.npz")}
            for s in SEQS for dt in (1, 4)},
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
