"""Phase 4 baselines: evaluate one unit (model x dataset x sequence x dt) and write its summary.

    python scripts/run_baselines.py --model zero  --dataset mvsec --sequence indoor_flying1 --dt 1
    python scripts/run_baselines.py --model cm    --dataset dsec  --sequence thun_00_a
    python scripts/run_baselines.py --model eraft --dataset mvsec --sequence outdoor_day1 --dt 4

Models: ``zero`` = M0a (arm zero_flow), ``cm`` = M0b (arm cm_baseline, parameters = argmin cell
of results/baselines/cm_tune.json, chosen on mvsec_tune only), ``eraft`` = R1 (arm eraft_mvsec20 on
MVSEC with the 20 Hz MVSEC checkpoint, eraft_dsec on DSEC with the DSEC checkpoint; tagged
reference_external, plus reference_seen_data on DSEC: E-RAFT trained on DSEC train sequences).

Writes ``results/baselines/<arm>/<dataset>_<sequence>_<dt k|100ms>.json`` (tier T1: real events,
real GT) and the per-window arrays to ``runs/phase4-baselines/npz/`` (gitignored). Skips a unit
whose JSON exists unless ``--force``. ``--max-windows N`` is a smoke run: it writes under
``runs/phase4-baselines/smoke/`` and never to results/.
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
from check_data import redact_home, sha256_file

from evflow.eval.datasets import dsec_unit, load_yaml, mvsec_unit
from evflow.eval.run import CMPredictor, ERAFTPredictor, ZeroPredictor, evaluate_windows
from evflow.eval.summary import summarise
from evflow.models.cm_baseline import CMParams
from evflow.models.eraft_runner import WarmStartRunner, load_eraft
from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "baselines"
RUNS = REPO / "runs" / "phase4-baselines"
CM_TUNE = RESULTS / "cm_tune.json"
MODEL_ID = {"zero": "M0a", "cm": "M0b", "eraft": "R1"}


def arm_of(model: str, dataset: str) -> str:
    if model == "eraft":
        return "eraft_mvsec20" if dataset == "mvsec" else "eraft_dsec"
    return {"zero": "zero_flow", "cm": "cm_baseline"}[model]


def frozen_cm_params() -> tuple[CMParams, dict]:
    if not CM_TUNE.exists():
        raise SystemExit(f"{CM_TUNE} missing: run scripts/tune_cm_baseline.py on mvsec_tune first")
    doc = json.loads(CM_TUNE.read_text())
    if doc.get("split") != "mvsec_tune":
        raise SystemExit(f"{CM_TUNE}: chosen on {doc.get('split')!r}, not mvsec_tune")
    params = CMParams.from_dict(doc["argmin"]["params"])
    return params, {"source": "results/baselines/cm_tune.json", "sha256": sha256_file(CM_TUNE),
                    "argmin_cell": doc["argmin"]["cell"]}


def build_unit(args, paths, device):
    ev_cfg = load_yaml("configs/eval_mvsec.yaml")
    er_cfg = load_yaml("configs/eraft.yaml")
    base = load_yaml("configs/evflow.yaml")
    if args.dataset == "mvsec":
        seq_cfg = ev_cfg["sequences"][args.sequence]
        region = ev_cfg["regions"][seq_cfg["region"]]
        windows, data, meta = mvsec_unit(paths, args.sequence, args.dt, split="mvsec_test",
                                         region=region)
        meta.update(region_name=seq_cfg["region"], region=region,
                    mask="valid_gt AND events_in_window (sparse) AND region",
                    gt="EV-FlowNet propagation of *_gt_flow_dist.npz (D-019)")
    else:
        delta = int(base["windows"]["dsec"]["delta_t_ms"]) * 1000
        windows, data, meta = dsec_unit(paths, args.sequence, delta_t_us=delta)
        meta.update(mask="valid_gt (PNG channel 3), full 640x480 rectified frame",
                    gt="DSEC forward flow PNG")
    extra = {"seed": None, "checkpoint_sha256": None, "representation": None, "tags": []}
    if args.model == "zero":
        pred = ZeroPredictor(data.height, data.width)
    elif args.model == "cm":
        params, prov = frozen_cm_params()
        pred = CMPredictor(params, data.height, data.width, device)
        extra.update(seed=0, representation="events (IWE)", params=params.to_dict(),
                     params_provenance=prov)
    else:
        arm = arm_of("eraft", args.dataset)
        ck = er_cfg["checkpoints"][arm]
        ck_path = Path(paths["checkpoints"][ck["key"]])
        model = load_eraft(ck_path, sha256=ck["sha256"], num_bins=er_cfg["num_bins"],
                           device=device)
        runner = WarmStartRunner(model, iters=er_cfg["iters"])
        crop = None
        span = None
        if args.dataset == "mvsec":
            r = ev_cfg["regions"][er_cfg["mvsec_input_crop"]]
            crop = (r["rows"][0], r["rows"][1], r["cols"][0], r["cols"][1])
        else:
            span = int(er_cfg["dsec"]["delta_t_ms"]) * 1000
        pred = ERAFTPredictor(runner, data.slicer, bins=er_cfg["num_bins"], height=data.height,
                              width=data.width, device=device, crop=crop, new_span_us=span)
        extra.update(checkpoint_sha256=ck["sha256"],
                     representation=f"voxel{er_cfg['num_bins']} (E-RAFT convention)",
                     tags=(["reference_external", "reference_seen_data"]
                           if args.dataset == "dsec" else ["reference_external"]),
                     params={"iters": er_cfg["iters"], "warm_start": True,
                             "input_crop": crop, "dsec_volume_span_us": span,
                             "source_commit": er_cfg["source_commit"]})
    return windows, data, meta, pred, extra, base["metrics"]["bootstrap"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(MODEL_ID), required=True)
    ap.add_argument("--dataset", choices=["mvsec", "dsec"], required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--dt", type=int, choices=[1, 4])
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if (args.dataset == "mvsec") != (args.dt is not None):
        ap.error("--dt is required for mvsec and not allowed for dsec")

    arm = arm_of(args.model, args.dataset)
    tag = f"dt{args.dt}" if args.dataset == "mvsec" else "100ms"
    stem = f"{args.dataset}_{args.sequence}_{tag}"
    smoke = args.max_windows is not None
    out_dir = (RUNS / "smoke" / arm) if smoke else (RESULTS / arm)
    out = out_dir / f"{stem}.json"
    if out.exists() and not args.force:
        print(f"skip: {out} exists")
        return 0

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    windows, data, meta, pred, extra, boot = build_unit(args, paths, args.device)
    if smoke:
        windows = windows[: args.max_windows]

    t0 = time.time()
    pw = evaluate_windows(windows, data, pred, log=lambda s: print(f"[{stem}] {s}", flush=True))
    wall = time.time() - t0
    summary = summarise(pw, block_seconds=float(boot["block_seconds"]),
                        resamples=int(boot["resamples"]), ci=float(boot["ci"]), seed=0)

    npz_dir = RUNS / ("smoke" if smoke else "npz") / arm
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"{stem}.npz"
    np.savez_compressed(npz, **pw)
    sha, dirty = git_sha(), git_dirty()
    doc = {
        "tier": "T1",
        "arm": arm,
        "model": MODEL_ID[args.model],
        "seed": extra["seed"],
        "checkpoint_sha256": extra["checkpoint_sha256"],
        "dataset": args.dataset,
        "sequence": args.sequence,
        **({"dt": args.dt} if args.dataset == "mvsec" else {"window_ms": 100}),
        "representation": extra["representation"],
        "tags": extra["tags"],
        **summary,
        "protocol": meta,
        **({"params": extra["params"]} if "params" in extra else {}),
        **({"params_provenance": extra["params_provenance"]}
           if "params_provenance" in extra else {}),
        "fwl_protocol": "interpolation=round, t_ref=1 (window end), events in the eval region",
        "per_window_npz": redact_home(npz),
        "per_window_npz_sha256": sha256_file(npz),
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "wall_time_s": round(wall, 1),
        "smoke": smoke,
        "git_sha": sha,
        "git_dirty": dirty,
        "created_by": "scripts/run_baselines.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    m = summary["metrics"]
    print(f"[{stem}] {arm}: EPE {m['epe']} FWL {m['fwl']} n={summary['n_windows']} "
          f"({wall:.0f} s) -> {redact_home(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
