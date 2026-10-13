"""Phase 8 part of ``results/TABLES.md`` (called by ``scripts/make_tables.py``; D-063).

"Deployment (Phase 8)": I-T1 ONNX parity on 200 fixed ``mvsec_test`` windows (T1: real events,
real GT; per run and provider, with the plan's tolerances and the verdict as computed by
``scripts/onnx_parity.py``), then J latency (hardware timing, no evidence tier: J1 representation
build, J2 network, J3 end-to-end). Reads committed JSON only: ``results/onnx_parity.json``,
``results/latency.json``; an absent file renders a "not produced" line.
"""

from __future__ import annotations

import json
from pathlib import Path


class TierError(ValueError):
    pass


def _table(head, rows):
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def _e(v) -> str:
    return "—" if v is None else f"{v:.2e}"


def _ms(p: dict | None) -> list[str]:
    if not p:
        return ["—"] * 4
    return [f"{p[k]:.3f}" for k in ("p50_ms", "p90_ms", "p99_ms", "max_ms")]


def parity_table(root: Path, sources: set) -> list[str]:
    rel = "results/onnx_parity.json"
    p = root / rel
    L = ["## Table I-T1 — ONNX parity (T1, mvsec_test, dt = 1)", ""]
    if not p.exists():
        return L + [f"Not produced yet (`{rel}`).", ""]
    doc = json.loads(p.read_text())
    sources.add(rel)
    tol = doc["tolerances"]
    n = doc["units"][0]["n_windows"]
    L += [f"{n} fixed windows ({doc['window_rule']}). Same input to both sides (representation "
          "built outside the graph); reference = torch fp32 eager on CUDA, TF32 off. "
          "max |Δflow| = largest vector difference over every pixel of every window (the "
          "gated number); masked = over the evaluation mask. EPE change = |EPE(ONNX) − "
          f"EPE(torch)| of the window mean. Tolerances: fp32 ≤ {tol['fp32_max_px']} px, "
          f"fp16 ≤ {tol['fp16_max_px']} px, EPE change ≤ {tol['epe_change_px']} px.", ""]
    rows = []
    for u in doc["units"]:
        if u.get("tier") != "T1":
            raise TierError(f"{rel}: unit {u.get('run_id')} is not T1")
        for label, r in u["providers"].items():
            m = r.get("metrics") or {}
            verdict = "PASS" if r["passed"] else ("FAIL" if r["status"] == "ok"
                                                 else f"FAIL ({r['status']})")
            rows.append(["T1", u["arm"], str(u["seed"]), label, r["graph_precision"],
                         _e(m.get("max_px")), _e(m.get("max_px_masked")),
                         _e(m.get("epe_change_px")), f"{r['tolerance_max_px']:g}", verdict])
        for label, r in (u.get("diagnostics") or {}).items():
            m = r["metrics"]
            rows.append(["T1", u["arm"], str(u["seed"]), f"{label} (diagnostic)", "fp16",
                         _e(m.get("max_px")), _e(m.get("max_px_masked")),
                         _e(m.get("epe_change_px")), "—", "not gated"])
    L += _table(["tier", "arm", "seed", "provider", "graph", "max |Δflow| (px)",
                 "masked max (px)", "EPE change (px)", "tol. max (px)", "verdict"], rows)
    L += ["", f"All rows pass: **{'yes' if doc['all_passed'] else 'no'}**."]
    if doc.get("non_conforming"):
        L += ["", f"**Non-conforming (D-064):** {', '.join(doc['non_conforming'])} fail the "
              "unchanged tolerance and carry no deployment claim; deployment claims are fp32 "
              f"only. Gate (every other row passes): "
              f"**{'yes' if doc.get('gate_passed') else 'no'}**."]
    L += [""] + sensitivity_lines(root, sources)
    return L


def sensitivity_lines(root: Path, sources: set) -> list[str]:
    rel = "results/fp16_sensitivity.json"
    p = root / rel
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    sources.add(rel)
    L = [f"**fp16 sensitivity** ({doc['split']} {doc['sequence']}, {doc['n_windows']} windows; "
         "diagnostic, not gated): max |Δflow| (px) vs fp32 when only the named part is fp16.",
         ""]
    rows = []
    for u in doc["units"]:
        v = u["variants"]
        mixed = {k: d["max_px"] for k, d in v.items() if k.startswith("mixed:")}
        best = min(mixed, key=mixed.get)
        rows.append([u["arm"], str(u["seed"]),
                     _e(v["input_rounded_fp16__fp32_compute"]["max_px"]),
                     _e(v["weights_rounded_fp16__fp32_compute"]["max_px"]),
                     _e(v["fp16_network"]["max_px"]),
                     _e(v["mixed:all_modules_fp32"]["max_px"]),
                     f"{_e(mixed[best])} ({best.removeprefix('mixed:')})"])
    L += _table(["arm", "seed", "input fp16", "weights fp16", "whole network fp16",
                 "inter-module activations fp16", "best mixed split"], rows)
    return L + [""]


def latency_tables(root: Path, sources: set) -> list[str]:
    rel = "results/latency.json"
    p = root / rel
    L = ["## Table J — latency (hardware timing; no evidence tier)", ""]
    if not p.exists():
        return L + [f"Not produced yet (`{rel}`).", ""]
    doc = json.loads(p.read_text())
    sources.add(rel)
    meth = doc["methodology"]
    env = doc["environment"]
    gpu = (env["ort_process"].get("gpu") or {}).get("name", "?")
    L += [f"{doc['hardware_note']} {doc['claims_note']} {doc['tier_note']}", "",
          f"**Accuracy.** {doc['accuracy_note']}" if doc.get("accuracy_note") else "",
          "",
          f"GPU {gpu}; batch {meth['batch_size']}; {meth['warmup']} warm-up, "
          f"{meth['timed']} timed iterations per cell; ms, p50 / p90 / p99 / max. Versions "
          "and GPU state: `results/latency.json` `environment`.", ""]
    L += ["### J1 — representation build (torch-CUDA, events on the GPU)", ""]
    rows = [[str(r["events"]), f"{r['width']}×{r['height']}", *_ms(r["build"]),
             f"{r['events_h2d']['p50_ms']:.3f}"] for r in doc["representation_build"]]
    L += _table(["events / window", "frame", "p50", "p90", "p99", "max",
                 "events H2D p50"], rows) + [""]
    L += ["### J2 — network forward (compute only; ORT H2D / D2H separate)", ""]
    rows = []
    for r in doc["network"]:
        if r.get("status", "ok") != "ok":
            rows.append([f"{r['width']}×{r['height']}", r["backend"], "—", "—", "—", "—",
                         "—", "—", f"failed: {r.get('error', '')[:60]}"])
            continue
        hp, wp = r["padded"]
        split = r.get("node_split")
        rows.append([f"{r['width']}×{r['height']} ({wp}×{hp})", r["backend"],
                     *_ms(r["compute"]),
                     f"{r['h2d']['p50_ms']:.3f} / {r['d2h']['p50_ms']:.3f}" if "h2d" in r
                     else "—",
                     f"{r['trt_engine_build_cold_s']:.1f}" if "trt_engine_build_cold_s" in r
                     else "—",
                     ", ".join(f"{k.removesuffix('ExecutionProvider')} {v}"
                               for k, v in split.items()) if split else "—"])
    L += _table(["input (padded)", "backend", "p50", "p90", "p99", "max", "H2D / D2H p50",
                 "TRT cold build (s)", "nodes per EP"], rows) + [""]
    L += ["### J3 — end-to-end (events → flow on the host)", "",
          "PyTorch rows are measured in one bracket; ORT rows are composed from stages timed "
          "in two processes (sum of i-th samples, tails assume independent stages; D-063) and "
          "include a voxel D2H + H2D round trip a single-process deployment would not pay; "
          "ort_cuda_fp16 rows omit the fp32 → fp16 cast. Σ stage max bounds the observed stage "
          "samples only.", ""]
    rows = [[str(r["events"]), f"{r['width']}×{r['height']}", r["backend"], r["how"],
             *_ms(r["total"]),
             f"{r['total']['sum_of_stage_max_ms']:.3f}" if "sum_of_stage_max_ms" in r["total"]
             else "—"] for r in doc["end_to_end"]]
    L += _table(["events / window", "frame", "backend", "how", "p50", "p90", "p99", "max",
                 "Σ stage max"], rows) + [""]
    return L


def deploy_part(root: Path, sources: set) -> list[str]:
    return (["# Deployment (Phase 8)", "",
             "ONNX export of M1 (opset 17), numerical parity (T1) and latency. Parity rows are "
             "T1; latency is a hardware measurement and carries no tier.", ""]
            + parity_table(root, sources) + latency_tables(root, sources))
