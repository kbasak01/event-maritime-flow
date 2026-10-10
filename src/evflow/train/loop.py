"""Train M1 for one MVSEC arm (IMPLEMENTATION_PLAN Phase 5 step 1-2).

    python -m evflow.train.loop --config configs/train_real_sup.yaml [--seed 2026]
        [--representation count] [--smoke] [--run-root runs/train]

bf16 autocast, AdamW, per-step cosine schedule, gradient clipping, random 256x256 crop +
horizontal flip (flow sign flip), polarity flip off (configs/evflow.yaml ``train``). Each epoch
draws ``epoch_samples`` windows with the configured dt mix, then evaluates the tune streams
(``train.tune_eval``). Selection: the epoch with the best tune metric on the arm's own selection
stream (EPE, min, for supervised arms; FWL, max, for self-supervised arms; dt = 1), on
``mvsec_tune`` only. Other streams (e.g. real tune events for v2e arms) are logged, never used.

Run directory ``<run-root>/<arm>_<rep>_s<seed>[<suffix>]/``: ``manifest.json``, ``metrics.csv``
(one row per epoch), ``last.pt`` (resume state), ``best.pt``, ``selection.json`` (rule, epoch,
value, checkpoint SHA-256) and ``DONE``. A run with ``DONE`` is skipped; an interrupted run
resumes from ``last.pt``.

``--smoke`` (make train-smoke): 8 fixed training windows, fixed centre crop, no augmentation;
trains until the eval-mode EPE on those windows (supervision mask) is below
``train.smoke.target_epe`` or ``max_steps`` is reached (exit 1). Writes under ``runs/smoke/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from evflow.losses.cm import CMLossParams, WindowEvents, cm_loss
from evflow.losses.supervised import masked_epe, multiscale_epe
from evflow.models.evflownet import EVFlowNet, n_params
from evflow.train import tune_eval
from evflow.train.data import (
    Augment,
    batch_inputs,
    cat_events,
    collate,
    in_channels,
    mixed_dt_weights,
)

REPO = Path(__file__).resolve().parents[3]
SELECTION_SPLIT = "mvsec_tune"
TUNE_METRICS = ("epe", "outlier_pct", "npe_1", "npe_3", "ae_deg", "fwl", "flow_mag")
DIAG_KEYS = ("flow_mag_events", "iwe_nonempty_ratio", "iwe_var_ratio", "abs_div")


# ---------------------------------------------------------------------------------------------
# config

def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Arm YAML merged over the file named by its ``inherit`` key (repo-relative)."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    files = [path]
    if "inherit" in doc:
        base_path = REPO / doc.pop("inherit")
        base = load_config(base_path)
        files = base.pop("_files") + files
        doc = _merge(base, doc)
    doc["_files"] = files
    return doc


def config_files_sha(cfg: dict) -> dict[str, str]:
    return {str(Path(f).resolve().relative_to(REPO)): hashlib.sha256(Path(f).read_bytes())
            .hexdigest() for f in cfg["_files"]}


def validate(cfg: dict) -> None:
    sel = cfg["selection"]
    if sel.get("split", SELECTION_SPLIT) != SELECTION_SPLIT or \
            cfg["train"]["selection_split"] != SELECTION_SPLIT:
        raise ValueError(f"selection must use {SELECTION_SPLIT} only (CLAUDE.md rule 4)")
    if cfg["loss"] not in ("supervised", "cm"):
        raise ValueError(f"loss must be supervised or cm, got {cfg['loss']!r}")
    if sel["mode"] not in ("min", "max") or sel["metric"] not in TUNE_METRICS:
        raise ValueError(f"bad selection rule {sel}")


def run_name(cfg: dict) -> str:
    rep = cfg["train"]["representation"]
    return f"{cfg['arm']}_{rep}_s{cfg['seed']}{cfg.get('run_suffix', '')}"


# ---------------------------------------------------------------------------------------------
# training

def seed_everything(seed: int) -> None:
    """Training randomness is torch only (sampler, crops, flips, CT draws in workers)."""
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict) -> EVFlowNet:
    return EVFlowNet(in_channels=in_channels(cfg["train"]["representation"]),
                     base_channels=int(cfg["model_m1"]["base_channels"]))


def _amp(cfg: dict, device) -> torch.dtype | None:
    if cfg["train"]["precision"] == "bf16" and torch.device(device).type == "cuda":
        return torch.bfloat16
    return None


def batch_loss(model, batch: dict, cfg: dict, device, amp_dtype, cm_params: CMLossParams | None
               ) -> tuple[torch.Tensor, dict[str, float]]:
    rep = cfg["train"]["representation"]
    h, w = batch["flow"].shape[-2:]
    inp = batch_inputs(batch, rep, h, w, device)
    with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                        enabled=amp_dtype is not None):
        preds = model(inp)
    preds = [p.float() for p in preds]
    gt = batch["flow"].to(device)
    mask = batch["mask"].to(device)
    stats = {}
    with torch.no_grad():
        stats["train_epe"] = float(masked_epe(preds[-1], gt, mask))   # logged for every arm
        stats["flow_mag"] = float(torch.linalg.vector_norm(preds[-1], dim=1).mean())
    if cfg["loss"] == "supervised":
        loss, parts = multiscale_epe(preds, gt, mask,
                                     list(cfg["losses"]["supervised"]["scale_weights"]))
        stats.update({f"epe_s{i}": v for i, v in enumerate(parts)})
    else:
        x, y, t, pol, b = cat_events(batch, device)
        keep = batch["ev_cm_keep"].to(device)
        tau = t.double() / batch["duration_us"].to(device)[b].double()
        evs = []
        for i in range(int(batch["n_events"].numel())):
            sel = keep & (b == i)
            evs.append(WindowEvents(x[sel], y[sel], tau[sel].float(), pol[sel]))
        loss, parts = cm_loss(preds[-1], evs, cm_params)
        stats.update(parts)
    return loss, stats


def train_epoch(model, loader, opt, sched, cfg: dict, device, cm_params, log=None,
                log_every: int = 200) -> dict[str, float]:
    model.train()
    amp_dtype = _amp(cfg, device)
    clip = float(cfg["train"]["grad_clip"])
    sums: dict[str, float] = {}
    gnorms = []
    n = 0
    for batch in loader:
        loss, stats = batch_loss(model, batch, cfg, device, amp_dtype, cm_params)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {sched.last_epoch}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        gnorms.append(float(gn))
        opt.step()
        sched.step()
        stats["train_loss"] = float(loss.detach())
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
        if log is not None and n % log_every == 0:
            log(f"  step {n}: loss {sums['train_loss'] / n:.4f}, grad norm {gnorms[-1]:.3f}")
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out.update(grad_norm_mean=float(np.mean(gnorms)) if gnorms else math.nan,
               grad_norm_max=float(np.max(gnorms)) if gnorms else math.nan, steps=n)
    return out


def make_loader(ds, cfg: dict, epoch: int, *, num_workers: int, epoch_samples: int):
    seed = int(cfg["seed"])
    fr = {int(k): float(v) for k, v in cfg["train"]["dt_fractions"].items()}
    weights = mixed_dt_weights(ds.dt_sizes, {dt: fr[dt] for dt in ds.dt_sizes})
    g = torch.Generator().manual_seed(seed * 7919 + epoch)
    sampler = torch.utils.data.WeightedRandomSampler(weights, epoch_samples, replacement=True,
                                                     generator=g)
    return torch.utils.data.DataLoader(
        ds, batch_size=int(cfg["train"]["batch_size"]), sampler=sampler, collate_fn=collate,
        num_workers=num_workers, drop_last=True, pin_memory=False,
        generator=torch.Generator().manual_seed(seed * 104729 + epoch))


def cosine_lambda(total_steps: int):
    def f(step: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / max(total_steps, 1)))
    return f


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flat_tune(stream: str, res: dict[int, dict]) -> dict[str, float]:
    row = {}
    for dt, r in sorted(res.items()):
        for k in TUNE_METRICS:
            row[f"{stream}_dt{dt}_{k}"] = r["metrics"].get(k)
        for k in DIAG_KEYS:
            if k in r["diag"]:
                row[f"{stream}_dt{dt}_diag_{k}"] = r["diag"][k]
    return row


def run_training(cfg: dict, run_dir: Path, train_ds, tune_ds: dict, *, device,
                 provenance: dict | None = None, num_workers: int = 0,
                 log=print) -> dict:
    """Train with resume; returns the selection record. ``tune_ds`` maps stream -> dataset and
    must contain ``cfg['events']['tune_select']``."""
    validate(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        log(f"{run_dir.name}: DONE present, skipping")
        return json.loads((run_dir / "selection.json").read_text())
    seed = int(cfg["seed"])
    seed_everything(seed)
    sel = cfg["selection"]
    sel_stream = cfg["events"]["tune_select"]
    streams = [sel_stream] + [s for s in cfg["events"].get("tune_log", []) if s != sel_stream]
    missing = [s for s in streams if s not in tune_ds]
    if missing:
        raise ValueError(f"tune streams {missing} not provided")
    region = cfg["train"]["tune_region"]
    if isinstance(region, str):
        region = yaml.safe_load((REPO / "configs" / "eval_mvsec.yaml").read_text())["regions"][
            region]
    model = build_model(cfg).to(device)
    epochs = int(cfg["train"]["epochs"])
    epoch_samples = (train_ds.dt_sizes[1] if cfg["train"]["epoch_samples"] == "dt1_windows"
                     else int(cfg["train"]["epoch_samples"]))
    steps_per_epoch = epoch_samples // int(cfg["train"]["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(epochs * steps_per_epoch))
    cm_params = CMLossParams.from_config(cfg["losses"]["cm"]) if cfg["loss"] == "cm" else None
    amp_dtype = _amp(cfg, device)
    better = (lambda a, b: a < b) if sel["mode"] == "min" else (lambda a, b: a > b)

    manifest_path = run_dir / "manifest.json"
    start_epoch, best = 0, {"epoch": None, "value": None}
    last = run_dir / "last.pt"
    if last.exists():
        # CPU: the RNG state must stay a CPU ByteTensor; load_state_dict moves the rest
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best = st["epoch"] + 1, st["best"]
        torch.set_rng_state(st["rng_torch"])
        random.setstate(st["rng_python"])
        log(f"{run_dir.name}: resuming at epoch {start_epoch}")
    else:
        from evflow.sim.v2e_runner import git_dirty, git_sha
        manifest = {
            "run_id": run_dir.name, "arm": cfg["arm"], "model": "M1",
            "representation": cfg["train"]["representation"], "seed": seed,
            "loss": cfg["loss"], "selection": {**sel, "stream": sel_stream,
                                               "split": SELECTION_SPLIT},
            "tune_streams": streams, "config_files": config_files_sha(cfg),
            "config_resolved": {k: v for k, v in cfg.items() if k != "_files"},
            "n_params": n_params(model), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "train_windows": train_ds.dt_sizes,
            "tune_windows": {s: d.dt_sizes for s, d in tune_ds.items()},
            "data": provenance or {}, "git_sha": git_sha(), "git_dirty": git_dirty(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if torch.device(device).type == "cuda"
                    else None),
            "argv": sys.argv, "created_by": "src/evflow/train/loop.py",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest_path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")

    csv_path = run_dir / "metrics.csv"
    rows = []
    if csv_path.exists() and start_epoch > 0:
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start_epoch]
    t_run = time.time()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        loader = make_loader(train_ds, cfg, epoch, num_workers=num_workers,
                             epoch_samples=epoch_samples)
        tr = train_epoch(model, loader, opt, sched, cfg, device, cm_params, log=log)
        t_train = time.time() - t0
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **tr,
               "train_s": round(t_train, 1)}
        for s in streams:
            res = tune_eval.evaluate(
                model, tune_ds[s], rep=cfg["train"]["representation"], region=region,
                device=device, batch_size=int(cfg["train"]["batch_size"]), amp_dtype=amp_dtype,
                diag_windows=int(cfg["train"]["tune_diag_windows"]) if s == sel_stream else 0,
                num_workers=num_workers)
            row.update(_flat_tune(s, res))
        row["epoch_s"] = round(time.time() - t0, 1)
        value = row.get(f"{sel_stream}_dt{sel['dt']}_{sel['metric']}")
        if value is None or not math.isfinite(value):
            raise FloatingPointError(f"selection metric is {value} at epoch {epoch}")
        if best["value"] is None or better(value, best["value"]):
            best = {"epoch": epoch, "value": value}
            torch.save({"model": model.state_dict(), "epoch": epoch, "arm": cfg["arm"],
                        "representation": cfg["train"]["representation"],
                        "in_channels": in_channels(cfg["train"]["representation"]),
                        "base_channels": int(cfg["model_m1"]["base_channels"]),
                        "seed": seed, "run_id": run_dir.name}, run_dir / "best.pt.tmp")
            os.replace(run_dir / "best.pt.tmp", run_dir / "best.pt")
        row["selected_so_far"] = best["epoch"]
        rows.append(row)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(csv_path.with_suffix(".csv.tmp"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        os.replace(csv_path.with_suffix(".csv.tmp"), csv_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "rng_torch": torch.get_rng_state(),
                    "rng_python": random.getstate()}, run_dir / "last.pt.tmp")
        os.replace(run_dir / "last.pt.tmp", last)
        log(f"{run_dir.name} epoch {epoch}: loss {tr['train_loss']:.4f} "
            f"{sel_stream} dt{sel['dt']} {sel['metric']} {value:.4f} "
            f"(best {best['value']:.4f} @ {best['epoch']}) {row['epoch_s']} s")

    selection = {"run_id": run_dir.name, "arm": cfg["arm"], "seed": seed,
                 "representation": cfg["train"]["representation"],
                 "rule": f"{sel['mode']} of {sel_stream} tune dt{sel['dt']} {sel['metric']} "
                         f"over epochs ({SELECTION_SPLIT} only)",
                 "split": SELECTION_SPLIT, "stream": sel_stream, "metric": sel["metric"],
                 "dt": sel["dt"], "mode": sel["mode"], "selected_epoch": best["epoch"],
                 "value": best["value"], "n_epochs": epochs, "checkpoint": "best.pt",
                 "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                 "wall_time_s_this_session": round(time.time() - t_run, 1),
                 "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    (run_dir / "DONE").write_text(selection["checkpoint_sha256"] + "\n")
    return selection


# ---------------------------------------------------------------------------------------------
# smoke: overfit 8 windows

def run_smoke(cfg: dict, train_ds, out_dir: Path, *, device, log=print) -> dict:
    if cfg["loss"] != "supervised":
        raise ValueError("--smoke overfits GT and needs a supervised arm")
    sm = cfg["train"]["smoke"]
    seed_everything(int(cfg["seed"]))
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sm["lr"]), weight_decay=0.0)
    batch = collate([train_ds[i] for i in range(len(train_ds))])
    amp_dtype = _amp(cfg, device)
    rep = cfg["train"]["representation"]
    h, w = batch["flow"].shape[-2:]
    inp = batch_inputs(batch, rep, h, w, device)
    gt, mask = batch["flow"].to(device), batch["mask"].to(device)
    weights = list(cfg["losses"]["supervised"]["scale_weights"])
    t0 = time.time()
    epe, step = math.inf, 0
    history = []
    for step in range(1, int(sm["max_steps"]) + 1):
        model.train()
        with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            preds = model(inp)
        loss, _ = multiscale_epe([p.float() for p in preds], gt, mask, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % int(sm["check_every"]) == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=torch.device(device).type,
                                                 dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(inp)[-1].float()
            epe = float(masked_epe(pred, gt, mask))
            history.append([step, epe])
            log(f"smoke step {step}: train loss {float(loss):.4f}, eval-mode EPE {epe:.4f} px")
            if epe < float(sm["target_epe"]):
                break
    from evflow.sim.v2e_runner import git_sha
    doc = {"kind": "train_smoke", "arm": cfg["arm"], "representation": rep,
           "n_windows": len(train_ds), "windows_k": batch["k"].tolist(),
           "windows_dt": batch["dt"].tolist(), "target_epe_px": float(sm["target_epe"]),
           "final_eval_epe_px": epe, "steps": step, "passed": epe < float(sm["target_epe"]),
           "history": history, "wall_s": round(time.time() - t0, 1), "device": str(device),
           "precision": cfg["train"]["precision"], "git_sha": git_sha(),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--representation", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-root", default=None, help="default: <paths.outputs.runs>/train")
    ap.add_argument("--suffix", default="", help="appended to the run id (sweep cells)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cm-smoothness", type=float, default=None)
    ap.add_argument("--cm-divergence", type=float, default=None)
    ap.add_argument("--max-train-windows", type=int, default=None, help="smoke-job runs only")
    ap.add_argument("--max-tune-windows", type=int, default=None, help="smoke-job runs only")
    ap.add_argument("--num-workers", type=int, default=None)
    a = ap.parse_args(argv)

    from evflow.train.data import load_mvsec
    cfg = load_config(a.config)
    cfg["seed"] = a.seed if a.seed is not None else int(cfg["seed_default"])
    if a.representation:
        cfg["train"]["representation"] = a.representation
    if a.epochs is not None:
        cfg["train"]["epochs"] = a.epochs
    if a.cm_smoothness is not None:
        cfg["losses"]["cm"]["smoothness_weight"] = a.cm_smoothness
    if a.cm_divergence is not None:
        cfg["losses"]["cm"]["divergence_penalty"] = a.cm_divergence
    cfg["run_suffix"] = a.suffix
    smoke_args = a.max_train_windows is not None or a.max_tune_windows is not None
    validate(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    tcfg = cfg["train"]
    nw = int(tcfg["num_workers"]) if a.num_workers is None else a.num_workers
    paths = yaml.safe_load((REPO / "configs" / "paths.local.yaml").read_text())
    runs = Path(paths["outputs"]["runs"])

    if a.smoke:
        sm = tcfg["smoke"]
        data = load_mvsec(train_source=cfg["events"]["train"], tune_streams=[], dts=[1],
                          warmup=int(tcfg["warmup_windows"]), hood_row=int(tcfg["hood_row"]),
                          augment=Augment(crop=tuple(cfg["model_m1"]["crop"]), hflip=False,
                                          fixed_crop=(2, 45)),
                          max_train_windows=int(sm["n_windows"]))
        doc = run_smoke(cfg, data.train, runs / "smoke" / f"{cfg['arm']}_{tcfg['representation']}",
                        device=device)
        print(f"train-smoke: {'PASS' if doc['passed'] else 'FAIL'} - eval-mode EPE "
              f"{doc['final_eval_epe_px']:.4f} px after {doc['steps']} steps "
              f"(target < {doc['target_epe_px']}), {doc['wall_s']} s")
        return 0 if doc["passed"] else 1

    root = Path(a.run_root) if a.run_root else runs / ("smoke-train" if smoke_args else "train")
    run_dir = root / run_name(cfg)
    if (run_dir / "DONE").exists():
        print(f"{run_dir.name}: DONE present, skipping")
        return 0
    streams = list(dict.fromkeys([cfg["events"]["tune_select"],
                                  *cfg["events"].get("tune_log", [])]))
    aug = tcfg["augment"]
    data = load_mvsec(
        train_source=cfg["events"]["train"], tune_streams=streams, dts=list(tcfg["dts"]),
        warmup=int(tcfg["warmup_windows"]), hood_row=int(tcfg["hood_row"]),
        augment=Augment(crop=tuple(cfg["model_m1"]["crop"]) if aug["random_crop"] else None,
                        hflip=bool(aug["hflip"]), polarity_flip=bool(aug["polarity_flip"])),
        max_train_windows=a.max_train_windows, max_tune_windows=a.max_tune_windows)
    sel = run_training(cfg, run_dir, data.train, data.tune, device=device,
                       provenance=data.provenance, num_workers=nw,
                       log=lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True))
    print(f"{run_dir.name}: selected epoch {sel['selected_epoch']} "
          f"({sel['metric']} {sel['value']:.4f}), checkpoint {sel['checkpoint_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
