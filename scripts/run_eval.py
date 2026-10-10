"""Evaluate a selected M1 checkpoint on a test unit (Phase 6 step 1; Phase 5 runs the gate units).

    python scripts/run_eval.py --run-dir runs/train/real_sup_voxel5_s1337 \
        --dataset mvsec --sequence indoor_flying1 --dt 1

The checkpoint is the run's ``best.pt`` as recorded in ``selection.json`` (selected on
``mvsec_tune`` only); its SHA-256 is verified before use. Test units only read the frozen
checkpoint: nothing is chosen here. Protocol, metrics and CIs as the Phase 4 baselines
(``eval/run.evaluate_windows``, ``eval/summary.summarise``; configs/eval_mvsec.yaml, D-019, D-041).
Input: the run's representation of the window's events on the full frame (zero-padded to a
multiple of 16, flow cropped back). Tier T1 (real events, real GT).

Writes ``results/eval/<arm>/<seed>/<dataset>_<sequence>_dt<k>[_<rep>].json`` (the suffix only
for non-default representations, i.e. the ablation) and the per-window arrays to
``runs/eval/npz/`` (gitignored). Skips an existing JSON unless ``--force``; ``--max-windows`` is a
smoke run written under ``runs/eval/smoke/`` only.
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
from evflow.eval.run import evaluate_windows
from evflow.eval.summary import summarise
from evflow.eval.windows import Window
from evflow.models.evflownet import EVFlowNet, pad_to_multiple, unpad
from evflow.sim.v2e_runner import git_dirty, git_sha
from evflow.train.data import make_input

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "eval"
RUNS = REPO / "runs" / "eval"


class NetPredictor:
    name = "m1"

    def __init__(self, model: EVFlowNet, rep: str, height: int, width: int, device) -> None:
        self.model, self.rep, self.h, self.w, self.device = model.eval(), rep, height, width, device

    @torch.no_grad()
    def __call__(self, w: Window, ev: dict) -> np.ndarray:
        x, y = np.asarray(ev["x"]), np.asarray(ev["y"])
        if np.issubdtype(x.dtype, np.floating) or np.issubdtype(y.dtype, np.floating):
            xt = torch.from_numpy(x.astype(np.float32))
            yt = torch.from_numpy(y.astype(np.float32))
        else:
            xt, yt = torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
        t = torch.from_numpy(np.asarray(ev["t"], np.int64) - w.t_start_us)
        p = torch.from_numpy(np.asarray(ev["p"]).astype(np.int64))
        inp = make_input(self.rep, xt.to(self.device), yt.to(self.device), t.to(self.device),
                         p.to(self.device), duration_us=w.t_end_us - w.t_start_us,
                         height=self.h, width=self.w)[None]
        inp, hw = pad_to_multiple(inp, 16)
        flow = unpad(self.model(inp)[-1].float(), hw)[0]
        return flow.double().cpu().numpy()


def load_run(run_dir: Path, device) -> tuple[EVFlowNet, dict, dict]:
    sel = json.loads((run_dir / "selection.json").read_text())
    man = json.loads((run_dir / "manifest.json").read_text())
    if sel.get("split") != "mvsec_tune":
        raise SystemExit(f"{run_dir}: selected on {sel.get('split')!r}, not mvsec_tune")
    ck = run_dir / sel["checkpoint"]
    if sha256_file(ck) != sel["checkpoint_sha256"]:
        raise SystemExit(f"{ck}: SHA-256 does not match selection.json")
    st = torch.load(ck, map_location=device, weights_only=False)
    model = EVFlowNet(in_channels=st["in_channels"], base_channels=st["base_channels"])
    model.load_state_dict(st["model"])
    return model.to(device).eval(), sel, man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", choices=["mvsec", "dsec"], required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--dt", type=int, choices=[1, 4])
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if (args.dataset == "mvsec") != (args.dt is not None):
        ap.error("--dt is required for mvsec and not allowed for dsec")

    run_dir = Path(args.run_dir)
    model, sel, man = load_run(run_dir, args.device)
    rep, arm, seed = man["representation"], man["arm"], man["seed"]
    default_rep = load_yaml("configs/evflow.yaml")["train"]["representation"]
    tag = f"dt{args.dt}" if args.dataset == "mvsec" else "100ms"
    stem = f"{args.dataset}_{args.sequence}_{tag}" + ("" if rep == default_rep else f"_{rep}")
    smoke = args.max_windows is not None
    out_dir = (RUNS / "smoke" / arm / str(seed)) if smoke else (RESULTS / arm / str(seed))
    out = out_dir / f"{stem}.json"
    if out.exists() and not args.force:
        print(f"skip: {out} exists")
        return 0

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    base = load_yaml("configs/evflow.yaml")
    boot = base["metrics"]["bootstrap"]
    if args.dataset == "mvsec":
        ev_cfg = load_yaml("configs/eval_mvsec.yaml")
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
                    gt="DSEC forward flow PNG",
                    resolution_note="zero-shot from 346x260 MVSEC training")
    if smoke:
        windows = windows[: args.max_windows]
    pred = NetPredictor(model, rep, data.height, data.width, args.device)

    t0 = time.time()
    pw = evaluate_windows(windows, data, pred, log=lambda s: print(f"[{stem}] {s}", flush=True))
    wall = time.time() - t0
    summary = summarise(pw, block_seconds=float(boot["block_seconds"]),
                        resamples=int(boot["resamples"]), ci=float(boot["ci"]), seed=0)
    npz_dir = RUNS / ("smoke" if smoke else "npz") / arm / str(seed)
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"{stem}.npz"
    np.savez_compressed(npz, **pw)
    doc = {
        "tier": "T1",
        "arm": arm,
        "model": "M1",
        "seed": seed,
        "checkpoint_sha256": sel["checkpoint_sha256"],
        "dataset": args.dataset,
        "sequence": args.sequence,
        **({"dt": args.dt} if args.dataset == "mvsec" else {"window_ms": 100}),
        "representation": rep,
        "tags": [],
        **summary,
        "protocol": meta,
        "run_id": man["run_id"],
        "selection": {k: sel[k] for k in ("rule", "split", "stream", "metric", "dt",
                                          "selected_epoch", "value")},
        "train_git_sha": man["git_sha"],
        "fwl_protocol": "interpolation=round, t_ref=1 (window end), events in the eval region",
        "per_window_npz": redact_home(npz),
        "per_window_npz_sha256": sha256_file(npz),
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "wall_time_s": round(wall, 1),
        "smoke": smoke,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "created_by": "scripts/run_eval.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    m = summary["metrics"]
    print(f"[{stem}] {arm} s{seed} {rep}: EPE {m['epe']} FWL {m['fwl']} "
          f"n={summary['n_windows']} ({wall:.0f} s) -> {redact_home(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
