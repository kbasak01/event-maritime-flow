# Phase 8 gate — ONNX export, parity, latency

Date: 2026-10-12 · final results at 5f567c5 (job run at 15e7143; resumed at db4f7c3, 13e34d8 and
5f567c5 for the diagnostics and summaries, all dirty=0) · Overall: **PASS with recorded fp16
non-conformance (D-064)** · CI green (run 38328046713 on f02904e).

History: first gate at 604b0a6 = FAIL (onnx-parity-auditor F-1, fp16 tolerance). The user chose
"Mixed-precision fix, re-test"; the tune-only sensitivity study (`results/fp16_sensitivity.json`)
showed no tested split brings real_sup under 5e-2 px; the user then chose, in this session on
2026-10-12, "Accept fp16 non-conforming (Recommended)" — D-064. The fp16 tolerance is unchanged
and the fp16 rows remain FAIL.

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
| 3 | Step 2: fp16 max \|Δflow\| ≤ 5e-2 px | **NOT MET — non-conformance recorded (D-064, user decision)** | ORT CUDA fp16: 9.34e-02 / 2.36e-01 px; TensorRT fp16: 9.57e-02 / 4.91e-01 px (real_sup / v2e_rand_sup); masked maxima also above (9.09e-02 … 4.28e-01); 85–163 of 200 windows exceed. Ungated diagnostic: torch eager fp16 itself reaches 5.59e-02 / 1.45e-01 px (`units[].diagnostics.torch_cuda_fp16`) |
| 4 | Step 2: EPE change ≤ 0.01 px | PASS | all six rows ≤ 1.10e-03 px (worst single window 0.018 px, not gated) |
| 5 | Step 2: 200 fixed `mvsec_test` windows | PASS | 50 per sequence, evenly spaced, `configs/evflow.yaml` `export.parity`; reference = Phase 6 path, per-window EPE equal to Phase 6 (max diff 0.0 / 4.4e-16) |
| 6 | Step 3: latency matrix + A4000 stand-in | PASS | `results/latency.json`, Tables J1–J3: build at 10⁴/10⁵/10⁶ events (346×260, 640×480); network 256², 352×272, 640×480 on eager fp32, ORT CUDA fp32/fp16, TensorRT fp16; 50/500 iterations, synchronised, p50/p90/p99/max; versions recorded; stand-in and no-sensor-latency sentences present |
| 7 | Step 4 / exit: auditor PASS | PASS (qualified) | first audit `VERDICT: FAIL` (F-1); re-audit after D-064 `VERDICT: PASS`, valid only under D-064 |
| 8 | Exit: both JSON committed | PASS | `git ls-files results/onnx_parity.json results/latency.json` |
| 9 | Results JSON keys / tiers | PASS | `onnx_parity.json` `kind` + `tier: T1`, each unit carries tier/arm/model/seed/checkpoint_sha256/dataset/sequences/dt/representation/n_windows (no `ci95`: deterministic comparison, not a sampled estimate); `latency.json` `kind: latency` with `tier_note`; `tests/test_results_have_tiers.py` passes |
| 10 | Commands | PASS | below |

## Commands

```
$ make lint
ruff check src tests scripts
All checks passed!
$ make test
393 passed, 29 deselected, 6 warnings in 70.14s (0:01:10)
$ make reproduce-tables
results/TABLES.md is current
```

`make test-data` not run: Phase 8 changed no `io/`, `sim/` or `eval/` code.

Job: `jobs/2026-10-12_export-bench.sh` — `runs/export-bench/job.log` (15e7143, dirty=0,
19:04–19:13) and `runs/export-bench/job_rerun_db4f7c3.log` (resume: torch fp16 diagnostic +
summaries + tables, dirty=0, 19:21–19:23).

## Agent verdicts (verbatim first line) — onnx-parity-auditor

### Re-audit (after db4f7c3, the sensitivity study and D-064, at 13e34d8)

> VERDICT: PASS

The auditor qualifies it: "This PASS holds only under D-064 … The plan criterion 'fp16 max
|Δflow| ≤ 5e-2 px' is still **NOT MET**", recorded as a user-approved non-conformance. It
re-computed every parity and latency number from the intermediates (0 mismatches), confirmed F-2
and M-1..M-5 resolved, and raised minors, all addressed at 5f567c5:
- N-1: bf16 claim now scripted (`bf16_autocast` in `results/fp16_sensitivity.json`); D-064 says
  "none of the 8 tested module splits" and discloses that a test-window scan informed them.
- N-2: wording scoped to real_sup (v2e_rand_sup passes with the encoder in fp32, 0.028 px);
  "tenfold" removed; the real_sup all-modules-fp32 anomaly stated.
- N-3: job logs are appended on resume; the overwritten parity-ORT / torch-fp16 console logs are
  disclosed in D-064 (sidecar JSONs keep status, providers, node split, environment).
- N-4: `fp16_sensitivity.json` has a `tier_note`, is a job unit with a log, and is committed.
- N-5: a non-conforming row still needs status ok, finite output and EPE change ≤ 0.01 px to count
  toward `gate_passed`.
Carry-forward from the auditor: Phase 9 adds the fp16 / D-064 entry to `docs/LIMITATIONS.md`;
with D-064 TensorRT has **no conforming row**, so Phase 10 must not present TensorRT-fp16 timings
as deployable (an optional TensorRT fp32 row would give TensorRT a conforming result).

### First audit (at 15e7143 results)

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
- The F-2 / M-x fixes were confirmed by the re-audit above.

## Open items

1. Phase 9 (Claude): `docs/LIMITATIONS.md` entry for the fp16 non-conformance (D-064) and the
   TensorRT no-conforming-row consequence; checklist F1 verdict "met-with-caveat (fp32) / unmet
   (fp16, D-064)".
2. Phase 10 (Claude): README deployment claims fp32 only; fp16 / TensorRT timings only with the
   parity caveat.
3. Note: `jobs/2026-10-12_export-bench.manifest.json` records config SHAs at preparation
   (15e7143); `configs/evflow.yaml` later gained `export.parity.non_conforming` (D-064), which
   affects only the summariser.
