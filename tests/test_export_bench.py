"""Latency harness (Phase 8 step 3, P1 methodology): methodology minimums, percentiles, CPU
fallback detection, sample composition and the environment record."""

import sys

import numpy as np
import pytest

from evflow.export import bench


def test_module_is_torch_free():
    # bench.py runs in the isolated ORT venv, which has no torch (D-063)
    src = open(bench.__file__).read()
    head = src.split("\ndef ", 1)[0]
    assert "import torch" not in head and "import onnxruntime" not in head


def test_minimums():
    assert bench.MIN_WARMUP_ITERS == 50 and bench.MIN_TIMED_ITERS == 500
    with pytest.raises(ValueError):
        bench.check_iters(49, 500)
    with pytest.raises(ValueError):
        bench.check_iters(50, 499)
    bench.check_iters(50, 500)


def test_percentiles():
    s = bench.percentiles(np.arange(1, 1001, dtype=np.float64))
    assert s["p50_ms"] == pytest.approx(500.5)
    assert s["p90_ms"] == pytest.approx(900.1)
    assert s["p99_ms"] == pytest.approx(990.01)
    assert s["max_ms"] == 1000.0
    assert s["n"] == 1000


def test_time_loop_counts_and_syncs():
    calls, syncs = [], []
    ms = bench.time_loop(lambda: calls.append(1), warmup=50, timed=500,
                         sync=lambda: syncs.append(1))
    assert len(calls) == 550 and ms.shape == (500,) and (ms >= 0).all()
    # one sync after warm-up, then one before and one after every timed call
    assert len(syncs) == 1 + 2 * 500
    with pytest.raises(ValueError):
        bench.time_loop(lambda: None, warmup=10, timed=500, sync=lambda: None)


def test_compose_samples():
    a = np.full(500, 1.0)
    b = np.arange(500, dtype=np.float64)
    c = bench.compose([a, b])
    assert c["p50_ms"] == pytest.approx(1.0 + 249.5)
    assert c["max_ms"] == 500.0
    assert c["sum_of_stage_max_ms"] == 500.0
    with pytest.raises(ValueError):
        bench.compose([a, b[:10]])


def test_resolve_providers_cpu_and_unknown():
    pytest.importorskip("onnxruntime")
    assert bench.resolve_providers(["CPUExecutionProvider"]) == ("CPUExecutionProvider",)
    with pytest.raises(ValueError):
        bench.resolve_providers(["NoSuchExecutionProvider"])


def test_resolve_providers_drops_cpu_fallback(monkeypatch):
    ort = pytest.importorskip("onnxruntime")

    class FallbackSession:
        def __init__(self, *a, providers=None, **k):
            self.req = providers

        def get_providers(self):
            return ["CPUExecutionProvider"]       # what ORT hands back when a library fails

    monkeypatch.setattr(ort, "InferenceSession", FallbackSession)
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert bench.resolve_providers(["CUDAExecutionProvider"]) == ()


def test_environment_info_keys():
    env = bench.environment_info()
    for k in ("python", "platform", "wsl2", "packages", "gpu"):
        assert k in env
    assert env["python"] == sys.version.split()[0]
    assert "numpy" in env["packages"]
