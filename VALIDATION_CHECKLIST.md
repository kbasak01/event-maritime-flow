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
- [x] A7 (critical) No test sequence used for calibration, CT range, epoch or model selection (grep of configs + run manifests). — Phase 9: met-with-caveat (33/33 `selection.json` and every calibration / sweep / fp16 study on `mvsec_tune`; D-064 scratch-run disclosure), `docs/LIMITATIONS.md`.

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
- [x] C3 Flow units = pixels over the window, forward; dt=1 and dt=4 never mixed in one mean. — Phase 6 gate: `docs/gates/phase-6.md` (criterion 9).
- [ ] C4 MVSEC eval windows, sparse mask, crop rule written in `configs/eval_mvsec.yaml` and quoted in README. — Phase 9: **unmet** (YAML complete; README is Phase 10), `docs/LIMITATIONS.md`.
- [x] C5 Phase 4 reproduction gate met (our metrics on E-RAFT flows = E-RAFT's own, ≤ 1e-3 px; EPE within 10 % of paper) or miss documented. — Phase 4 gate: `docs/gates/phase-4.md` (`results/baselines/eraft_reproduction.json`).
- [x] C6 Bootstrap CIs use moving blocks of 1 s; paired bootstrap for arm differences. — Phase 6 gate: `docs/gates/phase-6.md` (criterion 5; seed spread D-053).

## D. Training
- [x] D1 `train-smoke` overfits 8 windows to EPE < 0.1 px. — Phase 5 gate: `docs/gates/phase-5.md`.
- [x] D2 `real_sup` within 2× of EV-FlowNet-class EPE on `indoor_flying1` dt=1. — Phase 5 gate (D-043(h), D-046: within the 2.0 px gate, above zero flow; reported as a generalisation gap).
- [x] D3 No contrast-maximisation run shows event collapse (or it is excluded and listed). — Phase 5 gate (collapsed IWE-variance cells stopped and listed, D-047).
- [x] D4 Seeds 1337/2026/7 for the three core arms; spread reported. — `results/train/summary.json`, Phase 5 gate.
- [x] D5 Checkpoint SHA-256 recorded and referenced by every eval JSON. — Phase 5 gate (Phase 6 eval JSONs must keep this).
- [x] D6 Augmentation flips flow sign consistently with the image flip (unit test). — `tests/test_train_data.py::test_hflip_consistent_with_warping`.

## E. Claims
- [x] E1 (critical) Every README/docs number traces to a committed `results/*.json` (`claim-trace`). — Phase 9: met-with-caveat (claim-trace of LIMITATIONS, D-063, D-064; 2 findings fixed in D-063; README traced in Phase 10), `docs/LIMITATIONS.md`.
- [x] E2 (critical) Every table row carries a tier tag; no T3 row in a T1/T2 section. — Phase 6 gate (`results/TABLES.md`; re-checked at the Phase 7 gate with the T3 part, sim2real-study-runner PASS, `docs/gates/phase-7.md`).
- [x] E3 (critical) No HDR, glare-robustness or latency-advantage claim derived from v2e events. — Phase 9: met (grep; only disclaimers), `docs/LIMITATIONS.md`; README re-checked in Phase 10.
- [x] E4 "teacher-EPE" never called "EPE" or "accuracy"; FWL never called accuracy. — Phase 9: met, `docs/LIMITATIONS.md`.
- [x] E5 Maritime metrics reported per class (water/sky/obstacle) wherever LaRS labels exist. — Phase 7 gate (Tables F, G, H; `results/eval_maritime/summary.json`).
- [x] E6 Novelty statements match `IMPLEMENTATION_PLAN.md` §0 positioning (no claim to be first to train flow on synthetic events). — Phase 9: met-with-caveat (no novelty claim in docs; README in Phase 10), `docs/LIMITATIONS.md`.
- [x] E7 LaRS frame-rate assumption (if any) stated next to every LaRS number. — Phase 7 gate (T3 protocol, Table H, clutter figure; D-024). README to follow in Phase 10.

## F. Deployment
- [ ] F1 ONNX fp32 parity ≤ 1e-3 px (TF32 off); fp16 ≤ 5e-2 px; EPE change ≤ 0.01 px. — Phase 9: **unmet** (fp16 clause; fp32 and EPE change met; non-conformance accepted, D-064), `docs/LIMITATIONS.md`.
- [x] F2 Latency: ≥ 50 warm-up, ≥ 500 timed, synchronised; p50/p90/p99/max; representation build reported separately vs events per window. — Phase 8 gate; Phase 9: met (`results/latency.json` `methodology`), `docs/LIMITATIONS.md`.
- [x] F3 Hardware and software versions (GPU, driver, CUDA, ORT, TensorRT) in `results/latency.json`. — Phase 9: met (`environment.{torch_process,ort_process}`), `docs/LIMITATIONS.md`.
- [ ] F4 README states the A4000 is a stand-in for an embedded target. — Phase 9: **unmet** (README is Phase 10; sentence present in `results/latency.json` and Table J), `docs/LIMITATIONS.md`.

## G. Licences and publication
- [x] G1 (critical) No GPL code in `src/` or `tests/` (`test_no_gpl_imports.py` + reviewer). — Phase 9: met (test + repo-publication-reviewer dry run, `docs/gates/phase-9.md`), `docs/LIMITATIONS.md`.
- [x] G2 (critical) No dataset frames, event files or GT in git or release assets. — Phase 9: met for git incl. history (no release yet; release assets checked in Phase 10), `docs/LIMITATIONS.md`.
- [x] G3 Every figure derived from a CC BY-SA dataset is attributed and marked share-alike. — Phase 9: met-with-caveat (imagery panels attributed + share-alike, gitignored; committed EPE plots carry no imagery, README attributes in Phase 10), `docs/LIMITATIONS.md`.
- [x] G4 LaRS / SMD imagery appears only if `docs/LICENSES.md` permits. — Phase 7: no SMD imagery, LaRS GIFs gitignored; final check at publication (Phase 10); Phase 9: met-with-caveat (no SMD/LaRS imagery tracked), `docs/LIMITATIONS.md`.
- [x] G5 Vendored MIT code keeps its licence header and is listed in `docs/LICENSES.md`. — Phase 9: met, `docs/LIMITATIONS.md`.
- [x] G6 `make reproduce-tables` regenerates `results/TABLES.md` with no diff. — Phase 6 gate; enforced in CI (`.github/workflows/ci.yml`).
