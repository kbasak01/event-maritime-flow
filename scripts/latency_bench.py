"""Latency benchmark (Phase 8 step 3; plan §1.6, P1 methodology; D-063).

    python scripts/latency_bench.py torch                          # main venv (torch-CUDA)
    external/venvs/ort/bin/python scripts/latency_bench.py ort     # isolated ORT venv
    python scripts/latency_bench.py summarise                      # -> results/latency.json

Methodology in ``evflow.export.bench`` (>= 50 warm-up, >= 500 timed, synchronised brackets,
p50/p90/p99/max, batch 1). Settings from ``latency`` in configs/evflow.yaml.

``torch``: (a) representation build (the exported arm's voxel grid, ``train.data.make_input``)
on CUDA at 10^4 / 10^5 / 10^6 events per window, at 346x260 and 640x480, events resident on
the GPU; the host->device copy of the event arrays (int64 x, y, t, p, as the evaluation path
holds them) is timed separately. Events are synthetic: uniform x, y, p and sorted uniform t over
a fixed window, seeded — the build cost depends on the event count and frame size, not on scene
content. (b) PyTorch eager fp32 network (TF32 off, cuDNN benchmark off as in evaluation) at
256x256, 346x260 (padded to 352x272) and 640x480. (c) PyTorch end-to-end at 346x260, measured in
one bracket: event H2D + build + pad + network + flow D2H. (d) the stages the ORT end-to-end is
composed from: build + pad, and the D2H of the padded voxel grid.
``ort``: ORT CUDA EP fp32 / fp16 and TensorRT EP fp16 at the three network sizes with IO
binding (compute, H2D, D2H separately), the cold TensorRT engine build, and the node split.
``summarise``: one JSON with both processes' environment records. The ORT end-to-end runs in a
different process from the torch build (D-063: no single env holds torch-CUDA-12 and
ORT-CUDA-13), so it is **composed**: event H2D + build + pad + voxel D2H + ORT H2D + compute +
flow D2H, i-th samples summed; the sum of stage maxima is a hard upper bound. A single-process
deployment would skip the voxel D2H/H2D round trip, so the composed figure errs high.

The RTX A4000 stands in for an embedded target; nothing here measures or implies event-camera
sensor latency.
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
OUT_JSON = REPO / "results" / "latency.json"


def _cfg() -> dict:
    return yaml.safe_load((REPO / "configs/evflow.yaml").read_text())


def _paths() -> dict:
    return yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())


def _work(args) -> Path:
    if args.work_dir:
        return Path(args.work_dir)
    return Path(_paths()["outputs"]["runs"]) / "export" / "latency"


def _onnx_dir(args) -> Path:
    if args.onnx_dir:
        return Path(args.onnx_dir)
    return Path(_paths()["outputs"]["runs"]) / "export"


def _pad(h: int, w: int, m: int = 16) -> tuple[int, int]:
    return h + (-h) % m, w + (-w) % m


def _resolutions(cfg: dict, smoke: bool) -> list[list[int]]:
    r = cfg["latency"]["resolutions"]
    return [list(cfg["latency"]["end_to_end_resolution"])] if smoke else r


def _save(work: Path, name: str, doc: dict, samples: dict) -> None:
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / f"{name}_samples.npz", **samples)
    (work / f"{name}.json").write_text(json.dumps(doc, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------

def cmd_torch(args) -> int:
    import torch
    from run_eval import load_run

    from evflow.export.onnx import FinestFlow
    from evflow.models.evflownet import n_params, pad_to_multiple, unpad
    from evflow.sim.v2e_runner import git_dirty, git_sha
    from evflow.train.data import make_input

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg, paths = _cfg(), _paths()
    lc = cfg["latency"]
    warm, timed = int(lc["warmup"]), int(lc["timed"])
    rep, dur = lc["representation"], int(lc["synthetic_window_us"])
    dev = torch.device("cuda")
    sync = torch.cuda.synchronize
    run_id = cfg["export"]["runs"][0]
    model, sel, man = load_run(Path(paths["outputs"]["runs"]) / "train" / run_id, dev)
    if man["representation"] != rep:
        raise SystemExit(f"{run_id} uses {man['representation']}, latency config says {rep}")
    net = FinestFlow(model).eval()
    cin = int(model.enc[0][0].in_channels)
    rng = np.random.default_rng(int(lc["seed"]))
    events = [int(n) for n in lc["events_per_window"]]
    if args.smoke:
        events = events[:1]

    def synth(n: int, h: int, w: int) -> list[np.ndarray]:
        return [rng.integers(0, w, n, dtype=np.int64), rng.integers(0, h, n, dtype=np.int64),
                np.sort(rng.integers(0, dur, n, dtype=np.int64)),
                rng.integers(0, 2, n, dtype=np.int64)]

    doc: dict = {"kind": "latency_torch", "run_id": run_id,
                 "checkpoint_sha256": sel["checkpoint_sha256"], "n_params": n_params(model),
                 "representation": rep, "in_channels": cin, "synthetic_window_us": dur,
                 "warmup": warm, "timed": timed, "representation_build": [], "network": [],
                 "end_to_end": [], "stages_for_composition": []}
    samples: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        rep_res = lc["rep_resolutions"][:1] if args.smoke else lc["rep_resolutions"]
        for h, w in rep_res:
            for n in events:
                arrs = synth(n, h, w)
                h2d = bench.time_loop(lambda a=arrs: [torch.from_numpy(v).to(dev) for v in a],
                                      warmup=warm, timed=timed, sync=sync)
                xd, yd, td, pd = (torch.from_numpy(v).to(dev) for v in arrs)
                build = bench.time_loop(
                    lambda xd=xd, yd=yd, td=td, pd=pd, h=h, w=w: make_input(
                        rep, xd, yd, td, pd, duration_us=dur, height=h, width=w),
                    warmup=warm, timed=timed, sync=sync)
                key = f"rep_{h}x{w}_{n}"
                samples[f"{key}_build"], samples[f"{key}_h2d"] = build, h2d
                doc["representation_build"].append(
                    {"events": n, "height": h, "width": w, "device": "cuda",
                     "build": bench.percentiles(build), "events_h2d": bench.percentiles(h2d),
                     "events_dtype": "int64 x, y, t (us), p"})
                print(f"build {h}x{w} {n}: p50 {np.percentile(build, 50):.3f} ms", flush=True)

        for h, w in _resolutions(cfg, args.smoke):
            hp, wp = _pad(h, w)
            x = torch.randn(1, cin, hp, wp, device=dev,
                            generator=torch.Generator(device=dev).manual_seed(0))
            ms = bench.time_loop(lambda x=x: net(x), warmup=warm, timed=timed, sync=sync)
            samples[f"net_{hp}x{wp}_pytorch_eager_fp32"] = ms
            doc["network"].append({"height": h, "width": w, "padded": [hp, wp],
                                   "backend": "pytorch_eager_fp32", "precision": "fp32",
                                   "compute": bench.percentiles(ms),
                                   "settings": "TF32 off, cudnn.benchmark off, inference_mode"})
            print(f"eager {hp}x{wp}: p50 {np.percentile(ms, 50):.3f} ms", flush=True)

        h, w = lc["end_to_end_resolution"]
        for n in events:
            arrs = synth(n, h, w)

            def e2e(a=arrs, h=h, w=w):
                xd, yd, td, pd = (torch.from_numpy(v).to(dev) for v in a)
                inp, hw = pad_to_multiple(make_input(rep, xd, yd, td, pd, duration_us=dur,
                                                     height=h, width=w)[None], 16)
                return unpad(net(inp), hw).cpu()

            ms = bench.time_loop(e2e, warmup=warm, timed=timed, sync=sync)
            samples[f"e2e_{h}x{w}_{n}_pytorch_eager_fp32"] = ms
            doc["end_to_end"].append({"events": n, "height": h, "width": w,
                                      "backend": "pytorch_eager_fp32", "how": "measured",
                                      "stages": ["events H2D", "build", "pad", "network",
                                                 "flow D2H"],
                                      "total": bench.percentiles(ms)})
            print(f"e2e eager {n}: p50 {np.percentile(ms, 50):.3f} ms", flush=True)

            dev_ev = [torch.from_numpy(v).to(dev) for v in arrs]

            def build_pad(ev=dev_ev, h=h, w=w):
                return pad_to_multiple(make_input(rep, *ev, duration_us=dur, height=h,
                                                  width=w)[None], 16)[0]

            bp = bench.time_loop(build_pad, warmup=warm, timed=timed, sync=sync)
            vox = build_pad()
            d2h = bench.time_loop(lambda v=vox: v.cpu(), warmup=warm, timed=timed, sync=sync)
            h2d = bench.time_loop(lambda a=arrs: [torch.from_numpy(v).to(dev) for v in a],
                                  warmup=warm, timed=timed, sync=sync)
            for nm, s in (("events_h2d", h2d), ("build_pad", bp), ("voxel_d2h", d2h)):
                samples[f"stage_{h}x{w}_{n}_{nm}"] = s
            doc["stages_for_composition"].append(
                {"events": n, "height": h, "width": w,
                 "events_h2d": bench.percentiles(h2d), "build_pad": bench.percentiles(bp),
                 "voxel_d2h": bench.percentiles(d2h), "voxel_shape": list(vox.shape)})
    doc.update(environment=bench.environment_info(), git_sha=git_sha(), git_dirty=git_dirty(),
               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _save(_work(args), "torch", doc, samples)
    return 0


# ---------------------------------------------------------------------------------------------

def cmd_ort(args) -> int:
    cfg = _cfg()
    lc, pc = cfg["latency"], cfg["export"]["parity"]
    warm, timed = int(lc["warmup"]), int(lc["timed"])
    work, onnx_dir = _work(args), _onnx_dir(args)
    run_id = cfg["export"]["runs"][0]
    doc: dict = {"kind": "latency_ort", "run_id": run_id, "warmup": warm, "timed": timed,
                 "network": []}
    samples: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(lc["seed"]))
    for h, w in _resolutions(cfg, args.smoke):
        hp, wp = _pad(h, w)
        for label, provider, precision in pc["providers"]:
            fp16 = precision == "fp16"
            graph_fp16 = fp16 and provider == bench.CUDA
            onnx = onnx_dir / run_id / f"m1_{hp}x{wp}_{'fp16' if graph_fp16 else 'fp32'}.onnx"
            side = json.loads(onnx.with_suffix(".json").read_text())
            shape = side["input"]["shape"]
            x = rng.standard_normal(shape).astype(np.float16 if graph_fp16 else np.float32)
            row = {"height": h, "width": w, "padded": [hp, wp], "backend": label,
                   "provider": provider, "precision": precision,
                   "graph_precision": "fp16" if graph_fp16 else "fp32", "onnx": onnx.name,
                   "onnx_sha256": side["onnx_sha256"],
                   "checkpoint_sha256": side["checkpoint_sha256"],
                   "provider_options": bench.provider_options(provider, fp16=fp16,
                                                              cache_dir=None)}
            try:
                if provider not in bench.resolve_providers([provider]):
                    raise RuntimeError(f"{provider} does not construct a {provider} session")
                cache = None
                if provider == bench.TRT:
                    cache = work / "trt_cache" / f"{label}_{hp}x{wp}"
                    row["trt_engine_build_cold_s"] = round(
                        bench.engine_build_seconds(onnx, x, fp16=fp16, cache_dir=cache), 2)
                row["node_split"] = bench.node_split(onnx, provider, x, fp16=fp16,
                                                     cache_dir=cache)
                s = bench.ort_session(onnx, provider, fp16=fp16, cache_dir=cache)
                r = bench.bench_ort(s, x, warmup=warm, timed=timed)
                for k, v in r.pop("samples_ms").items():
                    samples[f"net_{hp}x{wp}_{label}_{k}"] = v
                row.update(status="ok", **r)
                print(f"{label} {hp}x{wp}: p50 {r['compute']['p50_ms']:.3f} ms "
                      f"{row['node_split']}", flush=True)
            except Exception as e:  # noqa: BLE001 - reported, never dropped
                row.update(status="failed", error=f"{type(e).__name__}: {e}")
                print(f"{label} {hp}x{wp}: FAILED {row['error']}", flush=True)
            doc["network"].append(row)
    doc.update(environment=bench.environment_info(),
               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _save(work, "ort", doc, samples)
    return 0


# ---------------------------------------------------------------------------------------------

def _accuracy_note() -> str:
    """Which backends passed parity (results/onnx_parity.json), so a fast row is not read as a
    deployable configuration when its numerics failed."""
    p = REPO / "results" / "onnx_parity.json"
    if not p.exists():
        return "Parity not yet measured (results/onnx_parity.json)."
    doc = json.loads(p.read_text())
    failed = sorted({lab for u in doc["units"] for lab, r in u["providers"].items()
                     if not r["passed"]})
    if not failed:
        return "Every timed ORT backend passed parity (results/onnx_parity.json)."
    return (f"Backends that failed parity (results/onnx_parity.json, Table I-T1): "
            f"{', '.join(failed)}; their timings are not a claim of a deployable "
            "configuration at the plan's tolerance.")


def cmd_summarise(args) -> int:
    from evflow.sim.v2e_runner import git_dirty, git_sha

    cfg = _cfg()
    lc = cfg["latency"]
    work = _work(args)
    td = json.loads((work / "torch.json").read_text())
    od = json.loads((work / "ort.json").read_text())
    ts, osm = np.load(work / "torch_samples.npz"), np.load(work / "ort_samples.npz")
    if od["run_id"] != td["run_id"]:
        raise SystemExit("torch and ORT benchmarks used different runs")
    for r in od["network"]:
        if r.get("checkpoint_sha256") != td["checkpoint_sha256"]:
            raise SystemExit(f"{r['onnx']}: built from another checkpoint")
    h, w = lc["end_to_end_resolution"]
    hp, wp = _pad(h, w)
    e2e = list(td["end_to_end"])
    for st in td["stages_for_composition"]:
        n = st["events"]
        base = [ts[f"stage_{h}x{w}_{n}_{k}"] for k in ("events_h2d", "build_pad", "voxel_d2h")]
        for r in od["network"]:
            if r["status"] != "ok" or r["padded"] != [hp, wp]:
                continue
            lab = r["backend"]
            parts = [osm[f"net_{hp}x{wp}_{lab}_{k}"] for k in ("h2d", "compute", "d2h")]
            e2e.append({"events": n, "height": h, "width": w, "backend": lab,
                        "how": "composed",
                        "stages": ["events H2D", "build + pad (torch-CUDA)", "voxel D2H",
                                   "ORT H2D", "ORT compute", "flow D2H"],
                        "total": bench.compose(base + parts)})
    network = list(td["network"]) + list(od["network"])
    doc = {
        "kind": "latency",
        "plan_step": "Phase 8 step 3",
        "tier_note": "Hardware timing of the M1 network and its input build; not a flow "
                     "metric, so no evidence tier applies. Inputs are synthetic (seeded).",
        "hardware_note": "RTX A4000 (desktop, WSL2) stands in for an embedded target; "
                         "embedded latency will differ.",
        "claims_note": "Nothing here measures or implies event-camera sensor latency or a "
                       "latency advantage of event cameras (CLAUDE.md rule 2).",
        "accuracy_note": _accuracy_note(),
        "methodology": {
            "warmup": td["warmup"], "timed": td["timed"], "batch_size": 1,
            "percentiles": ["p50", "p90", "p99", "max"],
            "timer": "time.perf_counter around each iteration; torch.cuda.synchronize before "
                     "and after (torch); IOBinding synchronize_inputs/outputs (ORT)",
            "ort_io": "IOBinding with device-resident input/output; H2D (fresh device copy "
                      "incl. allocation, an upper bound) and D2H timed separately",
            "provider_check": "a session must report the requested provider first "
                              "(evflow.export.bench.resolve_providers / ort_session); "
                              "node split from the ORT profile",
            "trt": "fp32 graph with trt_fp16_enable; cold engine build timed from an emptied "
                   "cache, then the cached engine is benchmarked",
            "end_to_end": "PyTorch: one measured bracket. ORT: composed from stages timed in "
                          "two processes and separate loops (torch-CUDA build, ORT network; "
                          "D-063), i-th samples summed; composed tails assume independent "
                          "stages; sum_of_stage_max_ms = upper bound over the observed stage "
                          "samples only (cross-process contention unmeasured). The ORT rows "
                          "include a voxel D2H + H2D round trip a single-process deployment "
                          "would not pay. ort_cuda_fp16 rows omit the fp32 -> fp16 cast of the "
                          "voxel grid (not timed)",
            "representation_events": "synthetic: uniform x, y, p; sorted uniform t over "
                                     f"{td['synthetic_window_us']} us; seed {lc['seed']}",
            "clocks": "not locked (WSL2 exposes no clock or persistence control); "
                      "environment.*.gpu is one nvidia-smi snapshot taken at the end of each "
                      "process (after its timed loops), not a per-cell record",
        },
        "model": {"arm_run": td["run_id"], "checkpoint_sha256": td["checkpoint_sha256"],
                  "n_params": td["n_params"], "representation": td["representation"],
                  "in_channels": td["in_channels"]},
        "representation_build": td["representation_build"],
        "network": network,
        "end_to_end": e2e,
        "environment": {"torch_process": td["environment"], "ort_process": od["environment"]},
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "bench_git_sha": td["git_sha"],
        "created_by": "scripts/latency_bench.py summarise",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["torch", "ort", "summarise"])
    ap.add_argument("--work-dir", help="default: <outputs.runs>/export/latency")
    ap.add_argument("--onnx-dir", help="default: <outputs.runs>/export")
    ap.add_argument("--smoke", action="store_true", help="one resolution, one event count")
    ap.add_argument("--out", help="summarise: write here instead of results/latency.json")
    args = ap.parse_args(argv)
    if args.out and Path(args.out).resolve().is_relative_to((REPO / "results").resolve()) \
            and Path(args.out).resolve() != OUT_JSON.resolve():
        ap.error("--out must not lie under results/")
    t0 = time.time()
    rc = {"torch": cmd_torch, "ort": cmd_ort, "summarise": cmd_summarise}[args.step](args)
    print(f"{args.step}: {time.time() - t0:.0f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
