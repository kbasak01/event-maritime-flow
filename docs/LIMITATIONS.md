# LIMITATIONS

Written in Phase 9 (2026-10-12, at 8928598 + this sweep), rows C4, E1, E6, E7, F4, G2–G4 updated in
Phase 10 (README, release): one row per VALIDATION_CHECKLIST.md
line. **C** = critical (blocks publication if unmet). Commands re-run for this sweep: `make lint`,
`make test` (393 passed), `make reproduce-tables` and a re-render in a clean worktree of HEAD
(no diff), `scripts/make_splits.py --check`; gate record `docs/gates/phase-9.md`.

| ID | Verdict (met / met-with-caveat / unmet) | Evidence | Note |
|---|---|---|---|
| A1 (C) | met | `docs/gates/phase-1.md` (event-data-auditor); `docs/gates/phase-3.md` criterion 5 (30 v2e MVSEC stores); `docs/gates/phase-7.md` (243 maritime stores); Phase 9: all 8 real MVSEC/DSEC stores re-checked by the eval-reproducibility-auditor (`docs/gates/phase-9.md`); store tests in `make test` | |
| A2 (C) | met | `docs/gates/phase-1.md`; `docs/DECISIONS.md` (MVSEC column order, µs, polarity) | |
| A3 | met-with-caveat | `docs/gates/phase-2.md`; D-022 | `scripts/h5_layout.py` stands in for `h5ls -v` (not installed) |
| A4 | met | `docs/gates/phase-1.md` | |
| A5 | met | `docs/gates/phase-1.md`; `configs/eval_mvsec.yaml` `pixel_space: raw_distorted` (D-015) | |
| A6 | met | `docs/gates/phase-2.md`; `scripts/make_splits.py --check` → "splits OK" (re-run in Phase 9); `splits/SHA256SUMS.txt` | |
| A7 (C) | met-with-caveat | all 33 `runs/**/selection.json` have `split: mvsec_tune`; `split: mvsec_tune` in `results/v2e_calibration.json`, `results/v2e_calibration_timing.json`, `results/baselines/cm_tune.json`, `results/train/cm_sweep.json`, `results/fp16_sensitivity.json`; `configs/evflow.yaml` `selection_split: mvsec_tune`; split guard `src/evflow/train/data.py` (`ALLOWED_SPLITS`); `docs/gates/phase-5.md` (event-data-auditor, selection leakage); clutter CT multipliers fixed a priori (D-054e); maritime arms selected on `mvsec_tune` (D-056) | Test-split names in configs appear only in evaluation-only settings (v2e test corpus, parity windows, figure panels). Caveat (D-064 disclosure): exploratory scratch runs on 50 `mvsec_test` parity windows informed which 8 precision splits the tune-only fp16 script tried; no precision configuration was selected or exported |
| B1 | met | `docs/gates/phase-2.md`; manifests in `results/v2e_*corpus.json` | |
| B2 | met | `docs/gates/phase-2.md` | |
| B3 | met | `docs/gates/phase-2.md`; re-calibration for timing on `mvsec_tune` (D-035, `results/v2e_calibration_timing.json`) | |
| B4 | met-with-caveat | `docs/gates/phase-3.md`; `results/v2e_alignment.json` | The 1-frame-shift control is not rejected on every group (see "What the alignment check can show" below) |
| B5 | met | `docs/gates/phase-3.md`; `results/fidelity/test_*.json` | |
| B6 | met | `docs/gates/phase-2.md`; `results/v2e_throughput.json`; D-026, D-054 | |
| C1 (C) | met | `docs/gates/phase-1.md`; oracle tests in `make test` | |
| C2 (C) | met | `docs/gates/phase-1.md`; FWL fixtures in `make test` | |
| C3 | met | `docs/gates/phase-6.md` (criterion 9) | |
| C4 | met | `configs/eval_mvsec.yaml` (windows, sparse mask, crop rule) quoted verbatim in `README.md` (generated block `eval-protocol`, Reproduce section; `make_tables.py --check` keeps it current) | Phase 10 |
| C5 | met | `docs/gates/phase-4.md`; `results/baselines/eraft_reproduction.json` | |
| C6 | met | `docs/gates/phase-6.md` (criterion 5); D-053 | |
| D1 | met | `docs/gates/phase-5.md` | |
| D2 | met-with-caveat | `docs/gates/phase-5.md`; D-043(h), D-046 | Within the 2× gate but above zero flow on `indoor_flying1` dt = 1; reported as a generalisation gap (Table A-T1) |
| D3 | met | `docs/gates/phase-5.md`; D-047 (collapsed cells stopped and listed) | |
| D4 | met | `results/train/summary.json`; `docs/gates/phase-5.md` | |
| D5 | met | `docs/gates/phase-5.md`; Phase 9 check: 403 eval JSONs of trained runs under `results/eval*/` carry `checkpoint_sha256` equal to the run's `selection.json` (0 missing, 0 mismatched) | |
| D6 | met | `tests/test_train_data.py::test_hflip_consistent_with_warping` (in `make test`) | |
| E1 (C) | met | `claim-trace` of `README.md` in Phase 10 (`docs/gates/phase-10.md`: 6 findings fixed, pass 2 CLEAN; every README number sits in a block generated from committed JSON, D-067, checked by `make reproduce-tables` in CI); Phase 9 traces of this file, D-063, D-064 | |
| E2 (C) | met | `docs/gates/phase-6.md`, `docs/gates/phase-7.md` (sim2real-study-runner PASS); `tests/test_results_have_tiers.py`; `make_tables._check_tier` / `tables_maritime.py` refuse mixed tiers | |
| E3 (C) | met | Phase 9 grep of `docs/`, `results/TABLES.md`, `scripts/`, `src/`, `configs/` for HDR / dynamic range / glare / low latency / latency advantage / microsecond: every hit is a disclaimer or a time unit (`results/TABLES.md` T3 tier note and Table J note, `scripts/fidelity_report.py`, `scripts/latency_bench.py`, `scripts/run_clutter.py`, `scripts/tables_maritime.py`, the `docs/CONVENTIONS.md` forbidden-word list, notes in `results/latency.json` and `results/maritime/clutter.json`) | README checked in the Phase 10 claim-trace: glare / dynamic range / latency occur only in disclaimers |
| E4 | met | T3 part of `results/TABLES.md` uses `teacher-EPE` only (plain "EPE" occurs only in Tables A–E and I, T1/T2); `tables_maritime.py` refuses the key `epe`; FWL/RSAT notes say "not accuracy" | |
| E5 | met | `docs/gates/phase-7.md` (Tables F, G, H; `results/eval_maritime/summary.json`) | |
| E6 | met | README Positioning section uses the plan §0 wording; no first / novel / state-of-the-art claim (Phase 10 claim-trace) | |
| E7 | met | README: LaRS frame rate stated as assumed 30 fps in the key findings, the T3 note box, the T3 protocol block, Table H and every T3 figure; MVSEC protocol stated before the key findings and quoted under Reproduce | |
| F1 | unmet | `results/onnx_parity.json` (`gate_passed: true`, `all_passed: false`, `non_conforming: [ort_cuda_fp16, ort_trt_fp16]`); `docs/gates/phase-8.md`; D-064 | fp32 (max 7.3e-05 px) and EPE change (≤ 1.1e-03 px) clauses met; the fp16 clause (≤ 5e-2 px) is not met (see the D-064 bullet below). Tolerance unchanged |
| F2 | met | `results/latency.json` `methodology` (warmup 50, timed 500, synchronised, p50/p90/p99/max); `representation_build` timed per event count (10⁴/10⁵/10⁶) separately from the network | |
| F3 | met | `results/latency.json` `environment.{torch_process,ort_process}`: GPU, driver, CUDA runtime, cuDNN, torch, onnxruntime-gpu, TensorRT versions | |
| F4 | met | `README.md` Deployment section and the generated key findings: "The RTX A4000 used here stands in for an embedded target"; also `results/latency.json` `hardware_note`, Table J, `results/figures/readme/latency.png` | Phase 10 |
| G1 (C) | met | `tests/test_no_gpl_imports.py` (scans `src/`, `tests/`, `scripts/`; in `make test`); `docs/LICENSES.md` (GPL repo not vendored, not imported); repo-publication-reviewer dry run (`docs/gates/phase-9.md`) | |
| G2 (C) | met | `git ls-files`: no h5 / npz / npy / bag / video / data files; committed images = metric plots in `results/figures/` and the README media in `results/figures/readme/` (derived visualisations, no raw frames, D-065, D-067); release assets (`docs/RELEASE.md`): checkpoints, fp32 ONNX, manifests, licence text only, `scripts/make_release.py` refuses data / image / video extensions | Phase 10 re-check |
| G3 | met | MVSEC/DSEC-derived README media (hero GIF, flow panels, fidelity figure, EPE and gap plots) are attributed and marked CC BY-SA 4.0 in their captions (`configs/evflow.yaml` `figures.attribution`; README captions) | Phase 10 |
| G4 | met | README LaRS media show derived content only, no source frame (D-065), marked CC BY-NC 4.0, attributed; no SMD imagery anywhere; `docs/LICENSES.md` LaRS and released-weights rows | Phase 10 |
| G5 | met | `src/evflow/models/eraft_ref/*.py` headers + `LICENSE` (MIT E-RAFT, BSD-3 RAFT); `tests/oracles/event_flow_metrics.py` MIT header; both listed in `docs/LICENSES.md` | |
| G6 | met | `make reproduce-tables` → "results/TABLES.md is current"; clean-worktree re-render, no diff; CI step in `.github/workflows/ci.yml` | |

### Unmet lines

- **Critical:** none.
- **Non-critical:** F1 (fp16 parity, accepted non-conformance D-064; no action, deployment
  claims stay fp32). C4 and F4 were closed in Phase 10 (README).

## Known limitations recorded before Phase 9

- **v2e timestamp pyramids (Phase 2, D-025).** v2e runs with a 0.75 ms timestamp cap, chosen in
  Phase 2 as the coarsest that honoured the 60 Hz photoreceptor lowpass check and kept after D-035
  (the lowpass is now off, so no lowpass check runs). Several events per pixel per interpolated
  frame are still spread in v2e's fixed pattern; temporal statistics of v2e events (inter-event
  intervals) carry this artefact and are discussed as such.
- **Symmetric contrast thresholds (Phase 2, D-027).** The plan's calibration grid ties
  pos_thres = neg_thres; the real DAVIS events of the tune slice are OFF-dominated, so the ON-fraction
  statistic cannot be matched by any grid cell.
- **Calibration slice (Phase 2, D-028).** The 32-cell calibration uses 3 × 4 s of `mvsec_tune`, not
  the full 60 s, for compute reasons.
- **Reduced corpus (Phase 2, D-026).** MVSEC arms train on 240 s of `outdoor_day2` (all arms the
  same windows); SMD (44.8 s train, 29.3 s test; D-054 superseded D-026's SMD cut), LaRS (110 train clips) and the clutter study (40 val
  clips) are subsets.
- **LaRS frame rate assumed (Phase 2, D-024).** v2e time constants for LaRS use an assumed 30 fps.
- **Timing-constrained v2e profile: correct event timing at a cost in event statistics (Phase 3,
  T2, D-035).** With the Phase 2 argmin (cutoff_hz 60) v2e events lagged the real events by half an
  APS frame on `mvsec_tune` (count-image NCC-vs-lag peak at +0.5 frame for cell c30,
  `results/v2e_calibration_timing.json`). The frozen profile is now the best lag-free cell, t10
  (thres 0.6, sigma 0.04, cutoff_hz 0): lag peak 0, tune objective (mean normalised W1) 0.701
  against 0.369 for c30 (same file). Turning the photoreceptor lowpass off is what removes the
  lag, so v2e here models no photoreceptor bandwidth; the distribution mismatch it brings is
  accepted and measured, not hidden.
- **The tune-calibrated v2e profile does not reproduce the test recordings' event statistics
  (Phase 3, T2, D-033).** Mean normalised W1 over the six §1.6 statistics, real vs v2e
  (`mvsec_cal`): 0.49 on `mvsec_tune` and 0.48 on the four train clips, but 1.47 (outdoor_day1) to
  2.49 (indoor_flying3) on the test sequences (`results/fidelity/summary.json`). On the indoor
  sequences v2e emits 2.72–2.89× the real event rate and its windows are half ON (0.50) where the
  real DAVIS windows are 0.71–0.74 ON; on outdoor_day1 v2e emits 0.82× the real rate, and the ON
  fraction (real 0.39, v2e 0.51) and the hot-pixel statistic (normalised W1 4.26; real sensor hot
  pixels, which v2e does not model) differ (`results/fidelity/test_*.json`); with the lowpass off
  the hot-pixel statistic also differs on tune and the train clips (normalised W1 1.50 and 1.33,
  `results/fidelity/summary.json`). These are measured
  statistics only; they were not used for any choice. Consequence for Phase 6: the T2 gap on test
  sequences includes this threshold/noise mismatch, which the CT-randomised arm is meant to
  address. Inter-event intervals of v2e carry the timestamp-pyramid artefact (above).
- **What the alignment check can show (Phase 3, T2, D-034, D-035).** Every `mvsec_cal` group
  passes the NCC threshold re-derived on tune (0.491063); indoor_flying2 by the smallest margin
  (median 0.498), and one of its two units, `test/test_indoor_flying2_p0`, is below it on its own
  (median 0.457; the gate is per group, D-034). The 5-frame-shift and x-mirror controls fall below it in every group
  (`results/v2e_alignment.json`). The 1-frame-shift control falls below it on tune and the three
  indoor sequences but not on outdoor_day1 or the train clips, so single-frame timing errors are
  not reliably caught by the threshold; the per-group NCC-vs-lag profile (`lag`) measures timing
  directly: its peak lies at +0.125 frame (one step of its 1/8-frame grid) on tune, train and the
  indoor sequences, and at +0.375 frame on outdoor_day1 (test sequence, reported only: median NCC
  0.675 at lag 0, 0.686 at the peak).
- **Maritime results are T3 only (Phase 7, D-054–D-058).** No ground-truth flow exists for the
  SMD or LaRS video. T3a numbers measure agreement with a frame-based RAFT teacher run on the
  same frames (the teacher is wrong where brightness constancy fails, e.g. on water and glint;
  its forward–backward mask removes only the inconsistent part), T3b numbers measure
  motion-compensation sharpness of the model's own events; neither is accuracy. The events are
  v2e simulations of 8-bit video: they inherit its dynamic range, motion blur, compression and
  30 fps sampling, so nothing in the maritime tables says anything about sun-glare or
  dynamic-range behaviour of a real event sensor (plan §0(1)).
- **Maritime geometry and timing (Phase 7, D-024, D-054).** Every maritime frame is resized by
  v2e to 640×360 without keeping the aspect ratio, so LaRS sources that are not 16:9 are
  stretched (per-unit `native_size` and `scale_xy` in `results/v2e_maritime_corpus.json`); teacher flow, class masks and predictions share
  that grid, and flow is reported in px per frame interval of that grid. LaRS frame rate is an
  assumption (30 fps), so v2e's time constants for LaRS and any per-second rate are assumed too.
- **Small maritime corpus (Phase 7, D-026, D-054).** SMD: every train video over [0, 5.6] s and
  every test video over [0, min(10 s, last frame)] (most videos are too short for D-026's 15 s
  segments); LaRS: 110 train clips and 40 evaluation clips (segment lengths per unit in
  `results/v2e_maritime_corpus.json` `units[].video_s`); the class-sliced and clutter results
  rest on the keyframe windows only (one per clip). Attribute levels with fewer than 3 clips are not
  reported, so rough seas and heavy reflections are mostly unrepresented in the clutter study.
  Some LaRS clips come from SMD footage; a thumbnail check finds none of the corpus's ones closer
  to any SMD on-board video than two different on-board videos are to each other
  (`results/maritime/lars_smd_overlap.json`, D-061), but it is a heuristic, not a proof.
- **Maritime arms selected on driving content (Phase 7, D-056).** CLAUDE.md rule 4 allows
  selection on `mvsec_tune` only, so `maritime_selfsup` and `maritime_teacher` pick their epoch
  on the v2e(mvsec_tune) stream, out of their training domain; the maritime arms are one seed each.
- **fp16 deployment is non-conforming; deployment claims are fp32 only (Phase 8, T1 parity,
  D-064).** On the 200 fixed `mvsec_test` dt = 1 parity windows, ORT CUDA EP fp32 matches PyTorch
  to max |Δflow| 7.3e-05 px (real_sup s1337) and 5.4e-05 px (v2e_rand_sup s7); every fp16 row
  misses the plan's 5e-2 px tolerance: ORT CUDA fp16 0.093 / 0.24 px, TensorRT EP fp16 0.096 /
  0.49 px (real_sup / v2e_rand_sup), also inside the evaluation mask, on 85–163 of 200 windows;
  the EPE change stays ≤ 1.1e-03 px in every row (`results/onnx_parity.json`
  `units[].providers.*.metrics`). Consequence: no TensorRT configuration conforms (the TensorRT EP
  row is fp16 only), so Table J's fp16 and TensorRT timings are not a deployable configuration at
  the plan's tolerance. A tune-only study found no tested fp32/fp16 module split that brings
  real_sup under the tolerance (`results/fp16_sensitivity.json`, D-064). The A4000 timings stand in
  for an embedded target and say nothing about event-sensor latency (`results/latency.json`).
- **Provenance of early results and of runs (Phase 9, eval-reproducibility-auditor m-2, m-3,
  m-5).** The Phase 2/3 result JSONs (`results/fidelity/*.json`, `v2e_alignment`,
  `v2e_calibration`, `v2e_calibration_timing`, `v2e_mvsec_corpus`, `v2e_mvsec_corpus_d032_lagged`,
  `v2e_maritime_corpus`, `v2e_throughput`, `lars_fps_check`) predate the `git_dirty` field; the
  corpus files record `checks.all_git_clean: true` and every `git_sha` is an ancestor of HEAD. The
  split manifests were generated at a dirty commit (`splits/README.md`); they regenerate
  byte-identically (A6). `splits/SHA256SUMS.txt` lists SHA-256 of the canonical JSON, not of the
  files, so it is not `sha256sum -c` input (`splits/README.md`). `runs/` is gitignored: the
  committed lineage stops at `results/train/*summary.json` `manifest_sha256`; split and
  corpus-plan SHAs per run live in `runs/*/manifest.json` (to be shipped or summarised in
  Phase 10).
