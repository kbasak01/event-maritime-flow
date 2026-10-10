# VALIDATION_CHECKLIST — event-maritime-flow

Phase 9 walks every line. Verdict ∈ {met, met-with-caveat, unmet} + evidence path, written to
`docs/LIMITATIONS.md`. "Critical" lines block publication if unmet.

## A. Data integrity and conventions
- [x] A1 (critical) Canonical store: `t` int64 µs monotonic, `p` ∈ {0,1}, `x<width`, `y<height`, `ms_to_idx` satisfies the DSEC definition — checked on every converted file (`event-data-auditor` report). — Phase 1 gate: `docs/gates/phase-1.md`.
- [x] A2 (critical) MVSEC event column order, time units and polarity encoding were verified against the file and recorded in `docs/DECISIONS.md`. — Phase 1 gate: `docs/gates/phase-1.md`.
- [x] A3 v2e `--dvs_h5` layout verified with `h5ls -v` and recorded before the converter was written. (`scripts/h5_layout.py` stands in for `h5ls`, not installed; D-022) — Phase 2 gate: `docs/gates/phase-2.md`.
- [x] A4 DSEC events rectified with `rectify_map.h5`; flow PNG decode round-trips; valid mask honoured. — Phase 1 gate: `docs/gates/phase-1.md`.
- [x] A5 MVSEC GT flow and events refer to the same (distorted/undistorted) pixel space; stated. — Phase 1 gate: `docs/gates/phase-1.md`.
- [x] A6 Split manifests regenerate byte-identically from the seed; SHA-256 committed. — Phase 2 gate: `docs/gates/phase-2.md`.
- [ ] A7 (critical) No test sequence used for calibration, CT range, epoch or model selection (grep of configs + run manifests).

## B. Simulator
- [x] B1 Every v2e output has a manifest (commit, argv, seed, input/output SHA, wall time). — Phase 2 gate: `docs/gates/phase-2.md`.
- [x] B2 `--input_frame_rate` set explicitly for every input; source documented. — Phase 2 gate: `docs/gates/phase-2.md`.
- [x] B3 Calibration used `mvsec_tune` only; objective and grid recorded; argmin reproducible. — Phase 2 gate: `docs/gates/phase-2.md`.
- [x] B4 v2e↔real alignment cross-correlation reported per sequence; threshold set before looking at test. — Phase 3 gate: `docs/gates/phase-3.md` (threshold re-derived on tune for the D-035 profile).
- [x] B5 Fidelity report (event statistics, real vs v2e) present for every test sequence. — Phase 3 gate: `docs/gates/phase-3.md`.
- [x] B6 Throughput measured and the corpus cuts it forced are documented. — Phase 2 gate: `docs/gates/phase-2.md`.

## C. Metrics
- [x] C1 (critical) EPE/%Out/NPE/AE match hand-computed fixtures and the vendored MIT oracle to 1e-6. — Phase 1 gate: `docs/gates/phase-1.md`.
- [x] C2 (critical) FWL = 1 for zero flow exactly; correct flow > wrong-sign flow on analytic fixtures. — Phase 1 gate: `docs/gates/phase-1.md`.
- [ ] C3 Flow units = pixels over the window, forward; dt=1 and dt=4 never mixed in one mean.
- [ ] C4 MVSEC eval windows, sparse mask, crop rule written in `configs/eval_mvsec.yaml` and quoted in README.
- [ ] C5 Phase 4 reproduction gate met (our metrics on E-RAFT flows = E-RAFT's own, ≤ 1e-3 px; EPE within 10 % of paper) or miss documented.
- [ ] C6 Bootstrap CIs use moving blocks of 1 s; paired bootstrap for arm differences.

## D. Training
- [ ] D1 `train-smoke` overfits 8 windows to EPE < 0.1 px.
- [ ] D2 `real_sup` within 2× of EV-FlowNet-class EPE on `indoor_flying1` dt=1.
- [ ] D3 No contrast-maximisation run shows event collapse (or it is excluded and listed).
- [ ] D4 Seeds 1337/2026/7 for the three core arms; spread reported.
- [ ] D5 Checkpoint SHA-256 recorded and referenced by every eval JSON.
- [ ] D6 Augmentation flips flow sign consistently with the image flip (unit test).

## E. Claims
- [ ] E1 (critical) Every README/docs number traces to a committed `results/*.json` (`claim-trace`).
- [ ] E2 (critical) Every table row carries a tier tag; no T3 row in a T1/T2 section.
- [ ] E3 (critical) No HDR, glare-robustness or latency-advantage claim derived from v2e events.
- [ ] E4 "teacher-EPE" never called "EPE" or "accuracy"; FWL never called accuracy.
- [ ] E5 Maritime metrics reported per class (water/sky/obstacle) wherever LaRS labels exist.
- [ ] E6 Novelty statements match `IMPLEMENTATION_PLAN.md` §0 positioning (no claim to be first to train flow on synthetic events).
- [ ] E7 LaRS frame-rate assumption (if any) stated next to every LaRS number.

## F. Deployment
- [ ] F1 ONNX fp32 parity ≤ 1e-3 px (TF32 off); fp16 ≤ 5e-2 px; EPE change ≤ 0.01 px.
- [ ] F2 Latency: ≥ 50 warm-up, ≥ 500 timed, synchronised; p50/p90/p99/max; representation build reported separately vs events per window.
- [ ] F3 Hardware and software versions (GPU, driver, CUDA, ORT, TensorRT) in `results/latency.json`.
- [ ] F4 README states the A4000 is a stand-in for an embedded target.

## G. Licences and publication
- [ ] G1 (critical) No GPL code in `src/` or `tests/` (`test_no_gpl_imports.py` + reviewer).
- [ ] G2 (critical) No dataset frames, event files or GT in git or release assets.
- [ ] G3 Every figure derived from a CC BY-SA dataset is attributed and marked share-alike.
- [ ] G4 LaRS / SMD imagery appears only if `docs/LICENSES.md` permits.
- [ ] G5 Vendored MIT code keeps its licence header and is listed in `docs/LICENSES.md`.
- [ ] G6 `make reproduce-tables` regenerates `results/TABLES.md` with no diff.
