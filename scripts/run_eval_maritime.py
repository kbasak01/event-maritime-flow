"""Evaluate one model on one T3 maritime unit (IMPLEMENTATION_PLAN Phase 7 step 4; D-057).

    python scripts/run_eval_maritime.py --model m1 \
        --run-dir runs/train/maritime_teacher_voxel5_s1337 --dataset lars --sequence lars_eval_kf
    python scripts/run_eval_maritime.py --model zero --dataset smd --sequence MVI_0790_VIS_OB

Models: ``m1`` (a trained run's selected checkpoint, SHA-verified), ``zero`` (M0a), ``cm`` (M0b,
frozen ``results/baselines/cm_tune.json`` parameters chosen on mvsec_tune), ``eraft`` (R1, the
DSEC checkpoint on the full 640x360 frame; its volumes are the predecessor window and the window,
warm-started along each clip; tag ``reference_external``). Nothing is chosen here.

One pass writes two summaries, never mixing tiers:
``results/eval_maritime/<arm>/<seed>/<dataset>_<sequence>_T3a.json`` (teacher-EPE family, RAFT
teacher, D-055) and ``..._T3b.json`` (FWL, RSAT); LaRS keyframe units carry ``by_class`` slices
(water / sky / obstacle). Per-window arrays go to ``runs/eval-maritime/npz/`` (gitignored).
Skips existing JSONs unless ``--force``; ``--max-windows`` writes a smoke run under
``runs/eval-maritime/smoke/`` only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from check_data import redact_home, sha256_file  # noqa: E402
from run_baselines import frozen_cm_params  # noqa: E402
from run_eval import NetPredictor, load_run  # noqa: E402

from evflow.eval import maritime as em  # noqa: E402
from evflow.eval.datasets import load_yaml  # noqa: E402
from evflow.eval.run import CMPredictor, ERAFTPredictor, ZeroPredictor  # noqa: E402
from evflow.sim.v2e_runner import git_dirty, git_sha  # noqa: E402

RESULTS = REPO / "results" / "eval_maritime"
RUNS = REPO / "runs" / "eval-maritime"
H, W = 360, 640
MODEL_ID = {"zero": "M0a", "cm": "M0b", "eraft": "R1", "m1": "M1"}
ARM = {"zero": "zero_flow", "cm": "cm_baseline", "eraft": "eraft_dsec"}


def build_predictor(args, device):
    """-> (predictor_for(segment), identity dict)."""
    if args.model == "zero":
        z = ZeroPredictor(H, W)
        return (lambda seg: z), {"arm": ARM["zero"], "seed": None, "checkpoint_sha256": None,
                                 "representation": None, "tags": []}
    if args.model == "cm":
        params, prov = frozen_cm_params()
        c = CMPredictor(params, H, W, device)
        return (lambda seg: c), {"arm": ARM["cm"], "seed": 0, "checkpoint_sha256": None,
                                 "representation": "events (IWE)", "tags": [],
                                 "params": params.to_dict(), "params_provenance": prov}
    if args.model == "eraft":
        from evflow.models.eraft_runner import WarmStartRunner, load_eraft
        paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
        er = load_yaml("configs/eraft.yaml")
        ck = er["checkpoints"]["eraft_dsec"]
        model = load_eraft(Path(paths["checkpoints"][ck["key"]]), sha256=ck["sha256"],
                           num_bins=er["num_bins"], device=device)
        runner = WarmStartRunner(model, iters=er["iters"])

        def for_seg(seg):
            return ERAFTPredictor(runner, seg.slicer, bins=er["num_bins"], height=H, width=W,
                                  device=device)
        return for_seg, {"arm": ARM["eraft"], "seed": None, "checkpoint_sha256": ck["sha256"],
                         "representation": f"voxel{er['num_bins']} (E-RAFT convention)",
                         "tags": ["reference_external"],
                         "params": {"iters": er["iters"], "warm_start": "along each clip",
                                    "volumes": "[prev window start, start) and [start, end)",
                                    "input": "full 640x360 frame (padded to /32 inside E-RAFT)",
                                    "note": "DSEC checkpoint trained on 100 ms volumes; "
                                            "maritime windows are one frame interval",
                                    "source_commit": er["source_commit"]}}
    model, sel, man = load_run(Path(args.run_dir), device)
    net = NetPredictor(model, man["representation"], H, W, device)
    return (lambda seg: net), {
        "arm": man["arm"], "seed": man["seed"], "checkpoint_sha256": sel["checkpoint_sha256"],
        "representation": man["representation"], "tags": [], "run_id": man["run_id"],
        "selection": {k: sel[k] for k in ("rule", "split", "stream", "metric", "dt",
                                          "selected_epoch", "value")},
        "train_git_sha": man["git_sha"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=list(MODEL_ID), required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--dataset", choices=["smd", "lars"], required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if (args.model == "m1") != (args.run_dir is not None):
        ap.error("--run-dir is required for --model m1 and only for it")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pred_for, ident = build_predictor(args, args.device)
    arm, seed = ident["arm"], ident["seed"]
    smoke = args.max_windows is not None
    out_dir = (RUNS / "smoke" if smoke else RESULTS) / arm / str(seed if seed is not None
                                                                 else "none")
    stem = f"{args.dataset}_{args.sequence}"
    outs = {t: out_dir / f"{stem}_{t}.json" for t in ("T3a", "T3b")}
    if all(o.exists() for o in outs.values()) and not args.force:
        print(f"skip: {outs['T3a']} and _T3b exist")
        return 0

    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    boot = load_yaml("configs/evflow.yaml")["metrics"]["bootstrap"]
    segs = em.segments(paths, args.dataset, args.sequence, size=(W, H))
    if smoke:
        left, keep = args.max_windows, []
        for s in segs:
            if left <= 0:
                break
            n = min(left, len(s.windows))
            s.windows, s.local_k, s.masks = s.windows[:n], s.local_k[:n], s.masks[:n]
            keep.append(s)
            left -= n
        segs = keep
    t0 = time.time()
    pw = em.evaluate(segs, pred_for, height=H, width=W,
                     log=lambda m: print(f"[{arm} {stem}] {m}", flush=True))
    wall = time.time() - t0
    npz_dir = (RUNS / ("smoke" if smoke else "npz") / arm / str(seed if seed is not None
                                                                else "none"))
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"{stem}.npz"
    np.savez_compressed(npz, **pw)
    cluster = args.dataset == "lars"
    sliced = args.sequence == "lars_eval_kf"
    kw = dict(cluster=cluster, block_seconds=float(boot["block_seconds"]),
              resamples=int(boot["resamples"]), ci=float(boot["ci"]), seed=0)
    profiles = load_yaml("configs/v2e_profiles.yaml")
    protocol = {
        "units": [s.key for s in segs], "store_sha256": {s.key: s.store_sha256 for s in segs},
        "teacher_sha256": {s.key: sha256_file(s.teacher) for s in segs},
        "end_clamped_us": {s.key: s.end_clamped_us for s in segs if s.end_clamped_us},
        "corpus": em.LARS_CORPUS if args.dataset == "lars" else em.SMD_CORPUS,
        "resolution": f"{W}x{H} (v2e output grid"
                      + ("; LaRS sources stretched, aspect not kept, D-054)" if cluster else ")"),
        "window": "one source-frame interval (frames k -> k+1); flow in px per frame interval",
        "frame_rate": (f"LaRS: assumed {profiles['lars']['assumed_fps']} fps (D-024), "
                       "px per frame interval is rate-free" if cluster
                       else "SMD: 30 fps (ffprobe r_frame_rate)"),
        "warmup_windows": em.WARMUP,
        "windows": ("the keyframe window (keyframe - 1 -> keyframe)" if sliced
                    else "every window after the warm-up"),
        "slices": "all = full frame" + ("; water/sky/obstacle = LaRS keyframe semantic mask, "
                                        "nearest-resized; ignore px in no class" if sliced else ""),
        "ci": ("cluster bootstrap over clips" if cluster
               else "moving-block bootstrap over windows, 1 s blocks"),
    }
    common = {
        "model": MODEL_ID[args.model], "arm": arm, "seed": seed,
        "checkpoint_sha256": ident["checkpoint_sha256"], "dataset": args.dataset,
        "sequence": args.sequence, "dt": 1, "dt_unit": "source frame interval",
        "representation": ident["representation"], "tags": ident["tags"],
        **{k: ident[k] for k in ("run_id", "selection", "train_git_sha", "params",
                                 "params_provenance") if k in ident},
        "n_windows_total": int(pw["index"].size),
        "per_window_npz": redact_home(npz), "per_window_npz_sha256": sha256_file(npz),
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "wall_time_s": round(wall, 1), "smoke": smoke, "git_sha": git_sha(),
        "git_dirty": git_dirty(), "created_by": "scripts/run_eval_maritime.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for tier, keys, note in (
            ("T3a", em.T3A_KEYS, "agreement with a frame-based RAFT teacher (teacher-EPE), "
                                 "not accuracy; mask = FB-consistent AND events-in-window AND "
                                 "slice (D-055, D-057)"),
            ("T3b", em.T3B_KEYS, "self-consistency (FWL > 1 better, RSAT < 1 better; round, "
                                 "t_ref = 1); not accuracy")):
        a = em.summarise_slice(pw, "all", keys, **kw)
        doc = {"tier": tier, "tier_note": note, **common, "metrics": a["metrics"],
               "ci95": a["ci95"], "n_windows": a["n_windows"],
               "n_windows_per_metric": a["n_windows_per_metric"]}
        if tier == "T3a":
            doc["mean_teacher_mask_px"] = a["mean_scored_px"]
        if sliced:
            doc["by_class"] = {s: em.summarise_slice(pw, s, keys, **kw)
                               for s in em.SLICES if s != "all"}
        doc["protocol"] = protocol
        doc["bootstrap"] = {"resamples": kw["resamples"], "ci": kw["ci"], "seed": 0,
                            "method": protocol["ci"]}
        out_dir.mkdir(parents=True, exist_ok=True)
        outs[tier].write_text(json.dumps(doc, indent=1) + "\n")
        print(f"[{arm} {stem}] {tier}: " + ", ".join(
            f"{k} {v:.4f}" for k, v in a["metrics"].items() if v is not None)
            + f" n={a['n_windows']} -> {redact_home(outs[tier])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
