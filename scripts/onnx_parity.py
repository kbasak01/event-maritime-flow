"""ONNX parity on 200 fixed mvsec_test windows (Phase 8 step 2; D-063).

    python scripts/onnx_parity.py ref                          # main venv (torch, data)
    external/venvs/ort/bin/python scripts/onnx_parity.py ort   # isolated ORT venv, no torch
    python scripts/onnx_parity.py summarise                    # -> results/onnx_parity.json

Windows: ``export.parity`` in configs/evflow.yaml — per test sequence, k windows evenly spaced
over the unit's dt=1 window list (``np.unique(np.linspace(0, n - 1, k).round())``), fixed before
any ONNX output existed. Evaluation-only data; nothing is chosen from it.

``ref``: real test events -> the run's representation -> zero-padded input, captured from
``run_eval.NetPredictor`` itself (so input and reference flow are bit-for-bit the Phase 6
evaluation path); torch fp32 on CUDA, TF32 off. GT and the evaluation mask (valid GT AND
events-in-window AND region, configs/eval_mvsec.yaml) are stored with the inputs. The
reference's per-window EPE is checked against the Phase 6 per-window arrays of the same run.
``ort``: the same inputs through ORT CUDA EP fp32 (TF32 off) and fp16 (fp16 graph) and the
TensorRT EP fp16 (fp32 graph, fp16 builder flag); each session must report its provider first,
and its node split is recorded. An unavailable or failing provider is recorded as such.
``summarise``: max |Δflow| (vector, px) over every pixel and over the mask, and the EPE change
(mean over windows of the per-window EPE), against the plan's tolerances.
Intermediate arrays live in ``<outputs.runs>/export/parity/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from evflow.export import bench

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "results" / "onnx_parity.json"


def _cfg() -> dict:
    return yaml.safe_load((REPO / "configs/evflow.yaml").read_text())


def _paths() -> dict:
    return yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())


def _work(args) -> Path:
    if args.work_dir:
        return Path(args.work_dir)
    return Path(_paths()["outputs"]["runs"]) / "export" / "parity"


def _onnx_dir(args) -> Path:
    if args.onnx_dir:
        return Path(args.onnx_dir)
    return Path(_paths()["outputs"]["runs"]) / "export"


def _subset(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, k).round().astype(np.int64))


# ---------------------------------------------------------------------------------------------
# ref (main venv)

def cmd_ref(args) -> int:
    import torch
    from run_eval import NetPredictor, load_run

    from evflow.eval.datasets import mvsec_unit
    from evflow.eval.windows import events_in_window, region_mask
    from evflow.flow.metrics import flow_metrics
    from evflow.sim.v2e_runner import git_dirty, git_sha

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, paths = _cfg(), _paths()
    pc = cfg["export"]["parity"]
    k = args.windows_per_sequence or int(pc["windows_per_sequence"])
    ev_cfg = yaml.safe_load((REPO / "configs/eval_mvsec.yaml").read_text())
    work = _work(args)
    work.mkdir(parents=True, exist_ok=True)
    runs_root = Path(paths["outputs"]["runs"])
    run_ids = cfg["export"]["runs"]
    device = args.device

    class Capture(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net, self.last = net, None

        def forward(self, x):
            self.last = x.detach().float().cpu().numpy()
            return self.net(x)

    models = {}
    for run_id in run_ids:
        model, sel, man = load_run(runs_root / "train" / run_id, device)
        models[run_id] = (Capture(model).eval(), sel, man)
    reps = {m[2]["representation"] for m in models.values()}
    if len(reps) != 1:
        raise SystemExit(f"parity runs must share one representation, got {reps}")
    rep = reps.pop()

    inputs, gts, masks, refs = [], [], [], {r: [] for r in run_ids}
    meta = {"sequence": [], "window_index": [], "t_start_us": [], "t_end_us": [],
            "n_events": []}
    units = []
    t0 = time.time()
    for seq in pc["sequences"]:
        sc = ev_cfg["sequences"][seq]
        region = ev_cfg["regions"][sc["region"]]
        windows, data, umeta = mvsec_unit(paths, seq, int(pc["dt"]), split=pc["split"],
                                          region=region)
        if umeta["split"] != "mvsec_test":
            raise SystemExit("parity windows must come from mvsec_test")
        pick = _subset(len(windows), k)
        units.append({"sequence": seq, "n_windows_unit": len(windows),
                      "positions": pick.tolist(), "region_name": sc["region"],
                      "split_manifest_sha256": umeta["split_manifest_sha256"]})
        crop = region_mask(region, data.height, data.width)
        preds = {r: NetPredictor(models[r][0], rep, data.height, data.width, device)
                 for r in run_ids}
        for j in pick:
            w = windows[int(j)]
            ev = data.slicer.get_events(w.t_start_us, w.t_end_us)
            x_first = None
            for r in run_ids:
                flow = preds[r](w, ev)
                x = models[r][0].last
                if x_first is None:
                    x_first = x
                elif not np.array_equal(x, x_first):
                    raise SystemExit(f"{r}: input differs from {run_ids[0]} at {seq} {w.index}")
                refs[r].append(flow.astype(np.float32))
            gt, valid = data.gt(w)
            eiw = events_in_window(ev["x"], ev["y"], data.height, data.width)
            inputs.append(x_first[0])
            gts.append(np.asarray(gt, np.float64))
            masks.append(np.asarray(valid, bool) & eiw & crop)
            meta["sequence"].append(seq)
            meta["window_index"].append(w.index)
            meta["t_start_us"].append(w.t_start_us)
            meta["t_end_us"].append(w.t_end_us)
            meta["n_events"].append(len(ev["t"]))
        print(f"ref {seq}: {len(pick)} windows ({time.time() - t0:.0f} s)", flush=True)

    np.savez_compressed(work / "inputs.npz", x=np.stack(inputs), gt=np.stack(gts),
                        mask=np.stack(masks), **{k_: np.asarray(v) for k_, v in meta.items()})
    doc = {"kind": "parity_ref", "representation": rep, "windows_per_sequence": k,
           "dt": int(pc["dt"]), "units": units, "n_windows": len(inputs),
           "input_shape": list(inputs[0].shape), "device": torch.cuda.get_device_name(0)
           if device.startswith("cuda") else "cpu", "tf32": False, "runs": {},
           "git_sha": git_sha(), "git_dirty": git_dirty(),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for r in run_ids:
        ref = np.stack(refs[r])
        np.savez_compressed(work / f"ref_{r}.npz", flow=ref)
        # cross-check against the Phase 6 per-window arrays of the same checkpoint
        _, sel, man = models[r]
        diffs = []
        for seq in pc["sequences"]:
            npz = runs_root / "eval" / "npz" / man["arm"] / str(man["seed"]) / \
                f"mvsec_{seq}_dt{int(pc['dt'])}.npz"
            if not npz.exists():
                continue
            p6 = np.load(npz)
            pos = {int(v): i for i, v in enumerate(p6["index"])}
            for i, (s_, wi) in enumerate(zip(meta["sequence"], meta["window_index"],
                                             strict=True)):
                if s_ != seq:
                    continue
                m = flow_metrics(torch.from_numpy(ref[i].astype(np.float64)),
                                 torch.from_numpy(gts[i]), valid_gt=torch.from_numpy(masks[i]))
                e6 = p6["epe"][pos[wi]]
                if np.isfinite(e6) != np.isfinite(m["epe"]):
                    diffs.append(float("inf"))
                elif np.isfinite(e6):
                    diffs.append(abs(m["epe"] - e6))
        doc["runs"][r] = {"checkpoint_sha256": sel["checkpoint_sha256"], "arm": man["arm"],
                          "seed": man["seed"],
                          "phase6_epe_max_abs_diff": float(max(diffs)) if diffs else None,
                          "phase6_windows_compared": len(diffs)}
        print(f"ref {r}: Phase 6 per-window EPE max |diff| "
              f"{doc['runs'][r]['phase6_epe_max_abs_diff']} over {len(diffs)} windows")
    (work / "ref.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------
# ort (isolated ORT venv)

def cmd_ort(args) -> int:
    cfg = _cfg()
    pc = cfg["export"]["parity"]
    work, onnx_dir = _work(args), _onnx_dir(args)
    x = np.load(work / "inputs.npz")["x"]
    h, w = 260, 346
    hp, wp = x.shape[-2:]
    for run_id in cfg["export"]["runs"]:
        for label, provider, precision in pc["providers"]:
            out = work / run_id / f"{label}.npz"
            proof = out.with_suffix(".json")
            if out.exists() and proof.exists() and not args.force:
                print(f"skip: {out} exists")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            fp16 = precision == "fp16"
            graph_fp16 = fp16 and provider == bench.CUDA
            onnx = onnx_dir / run_id / f"m1_{hp}x{wp}_{'fp16' if graph_fp16 else 'fp32'}.onnx"
            side = json.loads(onnx.with_suffix(".json").read_text())
            rec = {"kind": "parity_provider", "run_id": run_id, "label": label,
                   "provider": provider, "precision": precision, "onnx": onnx.name,
                   "onnx_sha256": side["onnx_sha256"],
                   "checkpoint_sha256": side["checkpoint_sha256"],
                   "graph_precision": "fp16" if graph_fp16 else "fp32",
                   "provider_options": bench.provider_options(provider, fp16=fp16,
                                                              cache_dir=None)}
            try:
                avail = bench.resolve_providers([provider])
                if provider not in avail:
                    raise RuntimeError(f"{provider} does not construct a {provider} session")
                cache = work / "trt_cache" / run_id / label if provider == bench.TRT else None
                dt = np.float16 if graph_fp16 else np.float32
                rec["node_split"] = bench.node_split(onnx, provider, x[:1].astype(dt),
                                                     fp16=fp16, cache_dir=cache)
                s = bench.ort_session(onnx, provider, fp16=fp16, cache_dir=cache)
                rec["session_providers"] = s.get_providers()
                name = s.get_inputs()[0].name
                flows = np.empty((x.shape[0], 2, h, w), np.float32)
                t0 = time.time()
                for i in range(x.shape[0]):
                    y = s.run(None, {name: x[i: i + 1].astype(dt)})[0]
                    flows[i] = y[0, :, :h, :w].astype(np.float32)
                rec.update(status="ok", wall_s=round(time.time() - t0, 2))
                np.savez_compressed(out, flow=flows)
            except Exception as e:  # noqa: BLE001 - reported, never dropped
                rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            rec["environment"] = bench.environment_info()
            proof.write_text(json.dumps(rec, indent=1) + "\n")
            print(f"ort {run_id} {label}: {rec['status']} {rec.get('node_split', '')}",
                  flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# summarise (main venv)

def _mean_epe(flows: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    from evflow.flow.metrics import flow_metrics

    return np.array([flow_metrics(torch.from_numpy(flows[i].astype(np.float64)),
                                  torch.from_numpy(gt[i]),
                                  valid_gt=torch.from_numpy(mask[i]))["epe"]
                     for i in range(flows.shape[0])])


def cmd_summarise(args) -> int:
    from evflow.export.onnx import parity_stats
    from evflow.sim.v2e_runner import git_dirty, git_sha

    cfg = _cfg()
    pc = cfg["export"]["parity"]
    tol = pc["tolerances"]
    work = _work(args)
    ref_doc = json.loads((work / "ref.json").read_text())
    data = np.load(work / "inputs.npz")
    gt, mask = data["gt"], data["mask"]
    units_out, all_pass = [], True
    for run_id in cfg["export"]["runs"]:
        rinfo = ref_doc["runs"][run_id]
        ref = np.load(work / f"ref_{run_id}.npz")["flow"]
        epe_ref = _mean_epe(ref, gt, mask)
        rows = {}
        for label, provider, precision in pc["providers"]:
            proof = json.loads((work / run_id / f"{label}.json").read_text())
            if proof["checkpoint_sha256"] != rinfo["checkpoint_sha256"]:
                raise SystemExit(f"{run_id}/{label}: ONNX built from another checkpoint")
            thr = tol["fp16_max_px"] if precision == "fp16" else tol["fp32_max_px"]
            row = {"provider": provider, "precision": precision,
                   "graph_precision": proof["graph_precision"], "onnx": proof["onnx"],
                   "onnx_sha256": proof["onnx_sha256"], "status": proof["status"],
                   "session_providers": proof.get("session_providers"),
                   "node_split": proof.get("node_split"),
                   "provider_options": proof["provider_options"],
                   "tolerance_max_px": thr, "tolerance_epe_change_px": tol["epe_change_px"]}
            if proof["status"] != "ok":
                row.update(error=proof.get("error"), passed=False)
                all_pass = False
                rows[label] = row
                continue
            test = np.load(work / run_id / f"{label}.npz")["flow"]
            st = parity_stats(ref, test, mask=mask)
            epe_test = _mean_epe(test, gt, mask)
            ok = np.isfinite(epe_ref) & np.isfinite(epe_test)
            d_epe = float(abs(np.mean(epe_test[ok]) - np.mean(epe_ref[ok])))
            passed = bool(st["max_px"] <= thr and d_epe <= tol["epe_change_px"])
            all_pass &= passed
            per_window = st.pop("per_window_max_px")
            row.update(metrics={**st, "epe_torch": float(np.mean(epe_ref[ok])),
                                "epe_onnx": float(np.mean(epe_test[ok])),
                                "epe_change_px": d_epe,
                                "max_window_epe_change_px": float(
                                    np.max(np.abs(epe_test[ok] - epe_ref[ok])))},
                       worst_windows=[{"sequence": str(data["sequence"][i]),
                                       "window_index": int(data["window_index"][i]),
                                       "max_px": per_window[i]}
                                      for i in np.argsort(per_window)[::-1][:3]],
                       passed=passed)
            rows[label] = row
        units_out.append({"tier": "T1", "arm": rinfo["arm"], "model": "M1",
                          "seed": rinfo["seed"], "run_id": run_id,
                          "checkpoint_sha256": rinfo["checkpoint_sha256"],
                          "dataset": "mvsec", "sequences": pc["sequences"],
                          "dt": int(pc["dt"]), "representation": ref_doc["representation"],
                          "n_windows": int(gt.shape[0]),
                          "n_windows_with_valid_px": int(np.isfinite(epe_ref).sum()),
                          "reference": "torch fp32 eager, CUDA, TF32 off "
                                       "(run_eval.NetPredictor path)",
                          "reference_vs_phase6_epe_max_abs_diff":
                              rinfo["phase6_epe_max_abs_diff"],
                          "reference_vs_phase6_windows": rinfo["phase6_windows_compared"],
                          "providers": rows})
    doc = {
        "kind": "onnx_parity",
        "tier": "T1",
        "plan_step": "Phase 8 step 2",
        "split": pc["split"],
        "window_rule": f"{ref_doc['windows_per_sequence']} per sequence, evenly spaced "
                       "(np.unique(np.linspace(0, n - 1, k).round())) over each mvsec_test "
                       "dt1 unit; fixed in configs/evflow.yaml export.parity before any ONNX "
                       "output",
        "windows": ref_doc["units"],
        "input_shape": ref_doc["input_shape"],
        "max_px_definition": "max over all pixels of all windows of |flow_onnx - flow_torch| "
                             "(2-vector norm, px), on the unpadded 260x346 frame; *_masked = "
                             "over the evaluation mask (valid GT AND events-in-window AND "
                             "region)",
        "epe_definition": "mean over windows (with >= 1 masked px) of the per-window mean EPE "
                          "vs MVSEC GT (flow.metrics.flow_metrics); change = |EPE_onnx - "
                          "EPE_torch|",
        "tolerances": tol,
        "all_passed": bool(all_pass),
        "units": units_out,
        "representation_build": "outside the ONNX graph; identical inputs fed to torch and "
                                "ONNX",
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "ref_git_sha": ref_doc["git_sha"],
        "created_by": "scripts/onnx_parity.py summarise",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    for u in units_out:
        for label, r in u["providers"].items():
            m = r.get("metrics", {})
            print(f"{u['run_id']} {label}: max {m.get('max_px')} px (tol "
                  f"{r['tolerance_max_px']}), dEPE {m.get('epe_change_px')} -> "
                  f"{'PASS' if r['passed'] else 'FAIL'}")
    print(f"all_passed={all_pass} -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["ref", "ort", "summarise"])
    ap.add_argument("--work-dir", help="default: <outputs.runs>/export/parity")
    ap.add_argument("--onnx-dir", help="default: <outputs.runs>/export")
    ap.add_argument("--windows-per-sequence", type=int, help="smoke override")
    ap.add_argument("--out", help="summarise: write here instead of results/onnx_parity.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.out and Path(args.out).resolve().is_relative_to((REPO / "results").resolve()) \
            and Path(args.out).resolve() != OUT_JSON.resolve():
        ap.error("--out must not lie under results/")
    return {"ref": cmd_ref, "ort": cmd_ort, "summarise": cmd_summarise}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
