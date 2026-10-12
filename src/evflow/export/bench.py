"""Latency harness (Phase 8 step 3; plan §1.6, P1 methodology, D-001, D-063).

Methodology, fixed before any measurement:
- at least 50 warm-up and 500 timed iterations; the helpers raise below that (a p99 over fewer
  than 500 samples is five points, not a tail estimate);
- every timed region is bracketed by a device synchronisation (``torch.cuda.synchronize`` for
  torch; ``IOBinding.synchronize_*`` for ONNX Runtime), wall clock ``time.perf_counter``;
- p50 / p90 / p99 / max reported, batch size 1;
- ONNX Runtime runs through an ``IOBinding`` with device-resident input and output, so compute is
  timed without the host<->device copies, which are timed separately (H2D, D2H);
- **a provider name is evidence only if a real session reports it.**
  ``get_available_providers()`` lists what the ORT binary was built with, not what loads on this
  machine; a provider library that fails to load yields a CPU session without raising (D-063:
  the main env's CUDA-13 ORT build next to torch's CUDA-12 libraries is exactly that case).
  ``resolve_providers`` builds a session per provider and keeps only those that report
  themselves first; ``ort_session`` refuses a session whose first provider is not the requested
  one; ``provider_node_counts`` reads ORT's profile to show how much of the graph TensorRT took;
- the TensorRT engine build is timed on its own, from an emptied cache (cold), never hidden in
  warm-up.

This module imports neither torch nor onnxruntime at module level: it runs both in the main env
(torch-CUDA representation build and eager network) and in the isolated ORT venv
``external/venvs/ort`` (no torch; D-063).
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

MIN_WARMUP_ITERS = 50
MIN_TIMED_ITERS = 500

TRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
DEVICE_PROVIDERS = frozenset({TRT, CUDA})

NVIDIA_SMI_FIELDS = ("name", "driver_version", "pstate", "clocks.sm", "clocks.max.sm",
                     "clocks.mem", "clocks.max.mem", "power.draw", "power.limit",
                     "persistence_mode", "temperature.gpu", "memory.total")

PACKAGES = ("numpy", "torch", "onnx", "onnxruntime-gpu", "onnxruntime", "tensorrt-cu13",
            "tensorrt-cu13-libs", "nvidia-cuda-runtime", "nvidia-cudnn-cu13", "nvidia-cublas",
            "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cublas-cu12")


def check_iters(warmup: int, timed: int) -> None:
    if warmup < MIN_WARMUP_ITERS:
        raise ValueError(f"warmup must be >= {MIN_WARMUP_ITERS}, got {warmup}")
    if timed < MIN_TIMED_ITERS:
        raise ValueError(f"timed iterations must be >= {MIN_TIMED_ITERS}, got {timed}")


def percentiles(ms: np.ndarray) -> dict:
    ms = np.asarray(ms, np.float64)
    return {"p50_ms": float(np.percentile(ms, 50)), "p90_ms": float(np.percentile(ms, 90)),
            "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()),
            "mean_ms": float(ms.mean()), "n": int(ms.size)}


def time_loop(fn: Callable[[], Any], *, warmup: int, timed: int,
              sync: Callable[[], Any]) -> np.ndarray:
    """Per-iteration wall time (ms) of ``fn`` with ``sync`` before and after every timed call."""
    check_iters(warmup, timed)
    for _ in range(warmup):
        fn()
    sync()
    out = np.empty(timed, np.float64)
    for i in range(timed):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out[i] = (time.perf_counter() - t0) * 1e3
    return out


def compose(stages: Sequence[np.ndarray]) -> dict:
    """End-to-end latency composed from stages timed separately (possibly in different
    processes): the i-th samples are summed, i.e. the stages are treated as independent and the
    sample order as arbitrary. ``sum_of_stage_max_ms`` is a hard upper bound on any one frame
    built from these stages."""
    arrs = [np.asarray(s, np.float64) for s in stages]
    if len({a.size for a in arrs}) != 1:
        raise ValueError(f"stage sample counts differ: {[a.size for a in arrs]}")
    total = np.sum(arrs, axis=0)
    return {**percentiles(total), "sum_of_stage_max_ms": float(sum(a.max() for a in arrs))}


# ---------------------------------------------------------------------------------------------
# ONNX Runtime

def preload() -> None:
    """Make ORT's CUDA/cuDNN (``ort.preload_dlls``) and TensorRT (``tensorrt_libs``, by ctypes)
    libraries loadable from site-packages; absence is not an error here, ``resolve_providers``
    decides availability."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    try:
        import tensorrt_libs
    except ImportError:
        return
    lib = Path(tensorrt_libs.__file__).parent
    for pattern in ("libnvinfer.so.*", "libnvonnxparser.so.*", "libnvinfer_plugin.so.*"):
        for cand in sorted(lib.glob(pattern)):
            if "builder_resource" in cand.name:
                continue
            try:
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def _probe_model() -> bytes:
    import onnx
    from onnx import TensorProto, helper

    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3],
                           np.linspace(-1, 1, 108, dtype=np.float32))
    g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
                          "probe", [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                                                  [1, 3, 16, 16])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                                         [1, 4, 16, 16])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = min(onnx.IR_VERSION, 9)
    return m.SerializeToString()


def resolve_providers(requested: Sequence[str], *, onnx_path: str | Path | None = None
                      ) -> tuple[str, ...]:
    """The subset of ``requested`` that really runs: a session built for it reports it first.
    Raises if a provider is not even built into this ORT binary."""
    import onnxruntime as ort

    preload()
    built = set(ort.get_available_providers())
    unknown = [p for p in requested if p not in built]
    if unknown:
        raise ValueError(f"onnxruntime was not built with {unknown}; built: {sorted(built)}")
    probe: Any = str(onnx_path) if onnx_path is not None else _probe_model()
    ok = []
    for p in requested:
        try:
            s = ort.InferenceSession(probe, providers=[p])
        except Exception:  # noqa: BLE001 - any construction failure = unavailable
            continue
        if s.get_providers()[:1] == [p]:
            ok.append(p)
    return tuple(ok)


def provider_options(provider: str, *, fp16: bool, cache_dir: Path | None) -> dict:
    if provider == CUDA:
        # TF32 off: the fp32 parity reference is computed with TF32 disabled (plan Phase 8)
        return {"device_id": 0, "use_tf32": 0, "cudnn_conv_algo_search": "EXHAUSTIVE"}
    if provider == TRT:
        opts: dict[str, Any] = {"device_id": 0, "trt_fp16_enable": bool(fp16)}
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            opts.update(trt_engine_cache_enable=True, trt_engine_cache_path=str(cache_dir))
        return opts
    return {}


def ort_session(path: str | Path, provider: str, *, fp16: bool = False,
                cache_dir: Path | None = None, profile_prefix: str | None = None):
    """Session on ``provider`` (TensorRT with the CUDA EP as fallback for nodes it rejects,
    reported by ``provider_node_counts``); raises if the session reports another provider
    first. ``fp16`` only matters for TensorRT (builder flag on the fp32 graph); the CUDA EP
    fp16 row runs the fp16 graph."""
    import onnxruntime as ort

    preload()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    provs = [provider] + ([CUDA] if provider == TRT else [])
    opts = [provider_options(p, fp16=fp16, cache_dir=cache_dir) for p in provs]
    s = ort.InferenceSession(str(path), so, providers=provs, provider_options=opts)
    got = s.get_providers()[0]
    if got != provider:
        raise RuntimeError(f"session for {provider} reports {got}: refusing a mislabelled row")
    return s


def provider_node_counts(session, feed: dict) -> dict[str, int]:
    """Nodes per execution provider from one profiled run (the session must have been built
    with ``profile_prefix``). A TensorRT subgraph shows as one fused node."""
    session.run(None, feed)
    prof = Path(session.end_profiling())
    events = json.loads(prof.read_text())
    prof.unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    for e in events:
        args = e.get("args") or {}
        if e.get("cat") == "Node" and args.get("provider") and e.get("name", "").endswith(
                "_kernel_time"):
            counts[args["provider"]] += 1
    return dict(counts)


def bench_ort(session, x: np.ndarray, *, warmup: int, timed: int) -> dict:
    """Compute latency with device-resident IO, plus per-iteration H2D (fresh device copy of
    ``x``; allocation included, so an upper bound on a reused staging buffer) and D2H of the
    outputs. Returns percentiles and the raw samples (ms)."""
    import onnxruntime as ort

    check_iters(warmup, timed)
    provider = session.get_providers()[0]
    dev = "cuda" if provider in DEVICE_PROVIDERS else "cpu"
    name = session.get_inputs()[0].name
    outs = [o.name for o in session.get_outputs()]
    b = session.io_binding()
    b.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(x, dev, 0))
    for o in outs:
        b.bind_output(o, dev, 0)
    for _ in range(warmup):
        session.run_with_iobinding(b)
    b.synchronize_outputs()
    comp, h2d, d2h = (np.empty(timed, np.float64) for _ in range(3))
    for i in range(timed):
        t0 = time.perf_counter()
        v = ort.OrtValue.ortvalue_from_numpy(x, dev, 0)
        b.bind_ortvalue_input(name, v)
        b.synchronize_inputs()
        h2d[i] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        session.run_with_iobinding(b)
        b.synchronize_outputs()
        comp[i] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        b.copy_outputs_to_cpu()
        d2h[i] = (time.perf_counter() - t0) * 1e3
    return {"provider": provider, "batch_size": int(x.shape[0]), "input_shape": list(x.shape),
            "input_dtype": str(x.dtype), "warmup": warmup, "timed": timed,
            "compute": percentiles(comp), "h2d": percentiles(h2d), "d2h": percentiles(d2h),
            "samples_ms": {"compute": comp, "h2d": h2d, "d2h": d2h}}


def engine_build_seconds(path: str | Path, x: np.ndarray, *, fp16: bool, cache_dir: Path
                         ) -> float:
    """Cold TensorRT engine build (session creation + first run) from an emptied cache."""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    s = ort_session(path, TRT, fp16=fp16, cache_dir=cache_dir)
    s.run(None, {s.get_inputs()[0].name: x})
    return time.perf_counter() - t0


def node_split(path: str | Path, provider: str, x: np.ndarray, *, fp16: bool,
               cache_dir: Path | None) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        s = ort_session(path, provider, fp16=fp16, cache_dir=cache_dir,
                        profile_prefix=str(Path(tmp) / "prof"))
        return provider_node_counts(s, {s.get_inputs()[0].name: x})


# ---------------------------------------------------------------------------------------------
# environment record (plan §1.6; checklist F3)

def _pkg_versions() -> dict[str, str]:
    from importlib import metadata

    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            continue
    return out


def _nvidia_smi() -> dict | None:
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(NVIDIA_SMI_FIELDS, vals, strict=False))


def environment_info() -> dict:
    """Interpreter, OS, package versions, GPU state (nvidia-smi) and, when importable, torch's
    CUDA/cuDNN and ORT/TensorRT versions. Clocks are not locked (WSL2 does not expose clock or
    persistence control); the state at measurement time is recorded instead."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wsl2": "microsoft" in platform.release().lower(),
        "cpu": platform.processor() or platform.machine(),
        "packages": _pkg_versions(),
        "gpu": _nvidia_smi(),
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS")},
    }
    try:
        import torch

        env["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(),
                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                        else None}
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        env["onnxruntime"] = {"version": ort.__version__,
                              "built_providers": ort.get_available_providers(),
                              "device": ort.get_device()}
    except ImportError:
        pass
    try:
        import tensorrt

        env["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    return env
