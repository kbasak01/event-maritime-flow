# Phase 8 gate — ONNX export, parity, latency

Date: 2026-10-12 · code at db4f7c3 (job run at 15e7143, resumed at db4f7c3 for the diagnostic and
summaries) · Overall: **FAIL** (fp16 parity; one user decision open). Not pushed.

## Gate text (IMPLEMENTATION_PLAN.md, Phase 8)

> 1. Export M1 (best arm) to ONNX opset 17, fp32 and fp16; the event-representation build stays
>    outside the graph (scatter-add in ONNX is fragile) and is benchmarked separately in torch-CUDA.
> 2. Parity on 200 fixed `mvsec_test` windows: max |Δflow| ≤ 1e-3 px fp32 (TF32 off), ≤ 5e-2 px fp16;
>    EPE change ≤ 0.01 px.
> 3. Latency (§1.6) for representation build at 10⁴/10⁵/10⁶ events, network at 256×256, 346×260
>    (padded to /16), 640×480; PyTorch eager, ORT CUDA EP, ORT TensorRT EP fp16. State that the A4000
>    stands in for an embedded target.
> 4. `onnx-parity-auditor`.
>
> **Exit gate:** `results/onnx_parity.json` and `results/latency.json` committed; auditor PASS.

Set-up decision: D-063 (checkpoints `real_sup_voxel5_s1337` + `v2e_rand_sup_voxel5_s7` by a
tune-only rule, user decision; isolated ORT venv; window rule; gated definition).

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Step 1: ONNX opset 17, fp32 + fp16, build outside the graph | PASS | 8 files in `runs/export/<run>/` with sidecars; opset 17, static shapes, ops Add/Concat/Constant/Conv/Relu/Resize; checkpoint SHAs match `selection.json` (auditor "Verified") |
| 2 | Step 2: fp32 max \|Δflow\| ≤ 1e-3 px (TF32 off) | PASS | `results/onnx_parity.json`: 7.32e-05 px (real_sup s1337), 5.42e-05 px (v2e_rand_sup s7), ORT CUDA EP, `use_tf32=0` |
| 3 | Step 2: fp16 max \|Δflow\| ≤ 5e-2 px | **FAIL** | ORT CUDA fp16: 9.34e-02 / 2.36e-01 px; TensorRT fp16: 9.57e-02 / 4.91e-01 px (real_sup / v2e_rand_sup); masked maxima also above (9.09e-02 … 4.28e-01); 85–163 of 200 windows exceed. Ungated diagnostic: torch eager fp16 itself reaches 5.59e-02 / 1.45e-01 px (`units[].diagnostics.torch_cuda_fp16`) |
| 4 | Step 2: EPE change ≤ 0.01 px | PASS | all six rows ≤ 1.10e-03 px (worst single window 0.018 px, not gated) |
| 5 | Step 2: 200 fixed `mvsec_test` windows | PASS | 50 per sequence, evenly spaced, `configs/evflow.yaml` `export.parity`; reference = Phase 6 path, per-window EPE equal to Phase 6 (max diff 0.0 / 4.4e-16) |
| 6 | Step 3: latency matrix + A4000 stand-in | PASS | `results/latency.json`, Tables J1–J3: build at 10⁴/10⁵/10⁶ events (346×260, 640×480); network 256², 352×272, 640×480 on eager fp32, ORT CUDA fp32/fp16, TensorRT fp16; 50/500 iterations, synchronised, p50/p90/p99/max; versions recorded; stand-in and no-sensor-latency sentences present |
| 7 | Step 4 / exit: auditor PASS | **FAIL** | onnx-parity-auditor `VERDICT: FAIL` (F-1 = criterion 3) |
| 8 | Exit: both JSON committed | PASS (at gate commit) | `git ls-files results/onnx_parity.json results/latency.json` |
| 9 | Results JSON keys / tiers | PASS | `onnx_parity.json` `kind` + `tier: T1`, each unit carries tier/arm/model/seed/checkpoint_sha256/dataset/sequences/dt/representation/n_windows (no `ci95`: deterministic comparison, not a sampled estimate); `latency.json` `kind: latency` with `tier_note`; `tests/test_results_have_tiers.py` passes |
| 10 | Commands | PASS | below |

## Commands

```
$ make lint
ruff check src tests scripts
All checks passed!
$ make test
392 passed, 29 deselected, 6 warnings in 60.57s (0:01:00)
$ make reproduce-tables
results/TABLES.md is current
```

`make test-data` not run: Phase 8 changed no `io/`, `sim/` or `eval/` code.

Job: `jobs/2026-10-12_export-bench.sh` — `runs/export-bench/job.log` (15e7143, dirty=0,
19:04–19:13) and `runs/export-bench/job_rerun_db4f7c3.log` (resume: torch fp16 diagnostic +
summaries + tables, dirty=0, 19:21–19:23).

## Agent verdict (verbatim first line) — onnx-parity-auditor

> VERDICT: FAIL

- **F-1 (blocking)**: fp16 parity misses 5e-2 px in all 4 fp16 rows, full frame and masked; EPE
  change within 0.01 px everywhere; fp32 passes; reporting honest (`all_passed: false`, Table
  I-T1 FAIL); gated definition matches D-063 (d) and is unchanged between smoke and full run.
  Fix needs a user decision: (a) record fp16 as non-conforming and limit deployment claims to fp32
  ORT CUDA, amending the gate wording in DECISIONS with user approval; or (b) fix fp16 numerics
  (mixed precision, e.g. decoder / flow heads in fp32) and re-run parity on the same 200 windows.
  The auditor notes that a post-hoc masked-pixel gate would not pass either and a percentile gate
  chosen now would defeat the pre-registration.
- **F-2 (major)**: D-063's smoke numbers did not match the cited artefact, and the torch-fp16
  claim had no script. **Addressed at db4f7c3**: numbers corrected to the job smoke JSON;
  `scripts/onnx_parity.py torch_fp16` adds a scripted, ungated diagnostic into
  `results/onnx_parity.json` and Table I-T1.
- **Minor** M-1 (composed end-to-end "hard upper bound" wording), M-2 (fp32→fp16 cast not in the
  composed fp16 row), M-3 (Table J did not flag fp16 parity failure), M-4 (cuDNN frontend errors
  in ORT CUDA fp16 sessions undisclosed), M-5 (GPU state = one end-of-process snapshot):
  **addressed at db4f7c3** by rewording / notes (`latency.json` `methodology`, `accuracy_note`,
  Table J, D-063). M-4's fallback convs are disclosed, not identified.
- The F-2 / M-x fixes post-date the audit and have not been re-audited; the re-audit happens
  with the F-1 resolution.

## Open items

1. **User**: decide F-1 — (a) accept fp16 as non-conforming (DECISIONS entry amending the gate
   for fp16; deployment claims fp32-only; fp16 timings kept with the accuracy note), or (b) Claude
   implements mixed-precision fp16 export (and TensorRT layer-precision constraints where ORT
   allows) and re-runs parity on the same 200 windows against the unchanged 5e-2 px tolerance.
2. **Claude**, after the decision: re-run onnx-parity-auditor, update this file, push, CI green.
