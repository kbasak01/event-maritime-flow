# Phase 5 gate — train the MVSEC arms (+ representation ablation)

Date: 2026-10-11 · Gate run by: Claude (`/phase-gate 5`) · Code commits: a2b1fa0 (Phase 5 code, job
prepared), f1967de (GPU resume fix, D-045), f9e996e (CM sweep stopped, D-047), 28f84d4 (average-
timestamp loss, rand-store end clamp, D-047/D-048), 7904922 (job relaunch commit; 13 of 17 runs and
all 8 sweep cells), fe8ef00 (guards from the selection-leakage audit) (+ this gate commit)

> **Exit gate:** all planned runs present with manifests; diagnostician TRUSTWORTHY on each (or the
> run is excluded with a reason); `real_sup` reaches EPE on `indoor_flying1` dt = 1 within 2× of the
> EV-FlowNet-class numbers in the literature (≈ 0.6–1.0 px) — if not, stop and fix before Phase 6.

Plan steps (IMPLEMENTATION_PLAN.md Phase 5): (1) `models/evflownet.py`, `losses/{supervised,cm}.py`,
`train/loop.py`, `make train-smoke` (overfit 8 windows: EPE < 0.1 px); (2) `/gpu-job` every MVSEC arm
of §1.4, selection on `mvsec_tune` (EPE supervised, FWL self-supervised), checkpoint SHA-256
recorded; (3) representation ablation (§1.5); (4) `training-run-diagnostician` on every run.

## Overall verdict: **PASS**

All 17 planned runs are present, selected on `mvsec_tune` only, and judged TRUSTWORTHY; none is
excluded. The `real_sup` gate threshold (D-043(h), confirmed by the user in D-046) is met. Tune
metrics below are selection evidence on `outdoor_day2` (T1 real events / T2 v2e events, real GT),
not test results; the only test numbers are the gate units in `results/eval/` (T1).

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Code: M1, supervised + CM losses with an event-collapse guard, training loop, arm configs | PASS | `src/evflow/models/evflownet.py` (13.3 M params, `tests/test_evflownet.py`), `src/evflow/losses/{supervised,cm}.py` (average-timestamp objective, multi-reference + divergence guard, D-047/D-048; `tests/test_losses_cm.py` incl. collapse-field tests), `src/evflow/train/{data,loop,tune_eval}.py`, `configs/train_{real_sup,v2e_fixed_sup,v2e_rand_sup,v2e_selfsup,real_selfsup}.yaml` |
| 2 | `make train-smoke` (8 windows, EPE < 0.1 px) | PASS | `train-smoke: PASS - eval-mode EPE 0.0830 px after 150 steps (target < 0.1), 11.2 s` |
| 3 | Every planned run present with manifest and checkpoint SHA-256 | PASS | `results/train/summary.json`: `17/17 runs done, 0 excluded`; each run's `manifest.json`, `selection.json` (checkpoint SHA verified against `best.pt` by the summariser and by every diagnostician) |
| 4 | Selection on `mvsec_tune` only (EPE / FWL), own-stream for v2e arms | PASS | `results/train/summary.json` (`selection.split = mvsec_tune` ×17, `used_for_selection` flags); event-data-auditor recomputed 17/17 selected epochs and 8/8 sweep cells from `metrics.csv` |
| 5 | CM loss weights chosen on tune only, collapse-checked | PASS | `results/train/cm_sweep.json` (8/8 cells eligible under D-048(c); D-044); the collapsed IWE-variance cells are stopped and kept apart (`runs/train-sweep-iwe-variance-d047/`, D-047) |
| 6 | Diagnostician TRUSTWORTHY on each run | PASS | 17/17 `VERDICT: TRUSTWORTHY` (table below; full reports in `docs/gates/phase-5-diagnostics/`) |
| 7 | `real_sup` on `indoor_flying1` dt = 1 within 2× of EV-FlowNet-class EPE (D-043(h): 3-seed mean ≤ 2.0 px) | PASS (with caveat) | `results/eval/real_sup/{1337,2026,7}/mvsec_indoor_flying1_dt1.json`: 1.4106 / 1.5084 / 1.2773 px, mean 1.3988 px ≤ 2.0. Caveat (D-046, user decision): the mean is above zero flow on the same windows (1.2786 px, `results/baselines/zero_flow/mvsec_indoor_flying1_dt1.json`); the s1337 checkpoint beats zero flow on `outdoor_day1` dt = 1 and `indoor_flying1` dt = 4 (`results/eval/real_sup/1337/`). Recorded as an outdoor→indoor generalisation gap, as for R1 (D-042); no training change from test numbers |
| 8 | Event-collapse: no CM run collapsed (or excluded and listed) | PASS | `real_selfsup`, `v2e_selfsup` and all 8 sweep cells pass the label-free checks; the two IWE-variance cells that collapsed were stopped after ≤ 2 epochs, never selected or used (D-047), listed here |
| 9 | Seeds 1337/2026/7 for the three core arms; spread reported | PASS | 9 runs in `results/train/summary.json`; three-seed tune spread per arm in the diagnostician reports (real_sup s7, v2e_fixed_sup s7, v2e_rand_sup s7) |
| 10 | Results JSON carry the `event-conventions` keys and come from clean trees | PASS | 5 eval files: tier T1, arm, model, seed, checkpoint_sha256, dataset, sequence, dt, representation, metrics, ci95, n_windows, git_sha, created_by, created_at; all `git_dirty = false`, none `smoke`; `summary.json` / `cm_sweep.json` declare `kind`, `git_dirty = false`; `tests/test_results_have_tiers.py` passes |
| 11 | No test split in any training/selection path | PASS | event-data-auditor (below); split guard `src/evflow/train/data.py` (`ALLOWED_SPLITS`, GT cache vs split manifest and warm-up, fe8ef00) |
| 12 | `make lint` | PASS | `All checks passed!` |
| 13 | `make test` | PASS | `314 passed, 22 deselected, 4 warnings in 337.69s` (slow host; 24 s on an idle machine) |
| 14 | `make test-data` (phase touched `eval/` via `scripts/run_eval.py` users and `train/`) | PASS | `22 passed, 314 deselected in 15.34s` |
| 15 | Artefacts committed | PASS | this gate commit: `results/train/{summary,cm_sweep}.json`, `results/eval/real_sup/**`, `docs/gates/phase-5*.md`, `jobs/2026-10-10_phase5-train.*` |

Checklist: **D1, D2, D3, D4, D5, D6** → ticked in `VALIDATION_CHECKLIST.md` (D5: every eval JSON
carries the checkpoint SHA of the run's `selection.json`; D6: `tests/test_train_data.py::test_hflip_consistent_with_warping`).

## Runs (selected epoch and selection value from `results/train/summary.json`)

| Run | Selection (own stream, tune dt1) | Diagnostician |
|---|---|---|
| real_sup voxel5 s1337 / s2026 / s7 | EPE, real (T1) | TRUSTWORTHY ×3 |
| v2e_fixed_sup voxel5 s1337 / s2026 / s7 | EPE, v2e (T2) | TRUSTWORTHY ×3 |
| v2e_rand_sup voxel5 s1337 / s2026 / s7 | EPE, v2e (T2) | TRUSTWORTHY ×3 |
| real_selfsup voxel5 s1337 | FWL, real (T1) | TRUSTWORTHY |
| v2e_selfsup voxel5 s1337 | FWL, v2e (T2) | TRUSTWORTHY |
| real_sup voxel15 / timesurface / count s1337 | EPE, real (T1) | TRUSTWORTHY ×3 |
| v2e_fixed_sup voxel15 / timesurface / count s1337 | EPE, v2e (T2) | TRUSTWORTHY ×3 |

## Agent verdicts

**training-run-diagnostician** (one review per run, read-only, no CUDA; full text in
`docs/gates/phase-5-diagnostics/<run>.md`): `VERDICT: TRUSTWORTHY` for all 17 runs. Findings worth
carrying forward (all non-blocking):

- real_sup s1337: resumed at epoch 24 after the D-045 GPU fault; LR, loss and tune curves continuous
  across the restart, nothing from the crashed partial epoch recorded. The manifest showed only the
  launch commit → resumes are now recorded in the manifest (fe8ef00).
- v2e_selfsup s1337 (medium): pre-clip gradient spikes up to ~1e6 with most late steps clipped; no
  damage to loss, tune FWL or logged real-tune EPE. Likely `tsum / (cnt + 1e-9)` at low-count
  pixels → D-049: floor low-count pixels before the next CM training (Phase 7).
- Self-supervised arms: no collapse; flow scale biased (over-predicted where no events, under on
  event pixels), typical of self-supervision; FWL-selected epochs sit on plateaus.
- Ablation (single seed each): voxel15 ≈ voxel5; count and timesurface worse on both tune streams;
  their real/v2e tune ratio is inside or slightly above the voxel5 seed range, so on tune the plan's
  §1.5 hypothesis (count less sensitive to v2e timestamp artefacts) finds no support. Phase 6
  reports this on the test sequences.
- Low: store `n_events` null in run provenance; `.v2e.json` sidecars not hashed (the store manifests
  are); time-surface tau hard-coded at window/3 while the config field is unused (D-050).

**event-data-auditor** (selection leakage and split integrity):

> VERDICT: PASS

No route by which a test split can reach training, selection or a GT cache; `make_splits.py --check`
reproduces the manifests; GT caches match their manifests and spot-recompute exactly; all selection
decisions recompute from tune-only data; test units read frozen, SHA-verified checkpoints and fed
nothing back (no supervised-path config/code change after the first test number). Minor findings
and actions:
1. The job stops on a test-number gate → D-043(h) now states a trip triggers diagnosis on tune only.
2. No command record for the D-046 extra units → command lines and reason added to D-046.
3. D-048(c) thresholds were set after viewing the collapsed cell incl. its logged tune EPE → disclosed in D-048.
4. GT-cache guard checked only split/dt attrs → now also the split-manifest SHA and warm-up (fe8ef00).
5. `freeze_cm_sweep.py` trusted `selection.json` → now asserts split and own stream (fe8ef00); output unchanged.
6. A smoke `run_eval.py --max-windows` used 20 `indoor_flying1` windows of a dev checkpoint (runs/ only, nothing selected) → open item: point smoke evals at a tune or fixture unit.
7. Resume commit not recorded in the manifest → fixed (fe8ef00).
8. DECISIONS formatting (D-042/D-043 merged, D-044 out of order) → fixed.

## Incidents during the phase (all recorded in DECISIONS)

- D-045: CUDA "unspecified launch failure" with two training processes on the GPU; WSL2 driver
  stalls; resume fix; concurrency 1.
- D-047: IWE-variance CM loss collapsed on real MVSEC within 2 epochs; sweep stopped; objective
  switched to average timestamp (user decision).
- D-048: rand-store end clamp; a second CUDA process stalls the driver on this workstation (one GPU
  process at a time); stricter label-free CM eligibility; average-timestamp pilot.

## Open items

| Item | Who |
|---|---|
| Phase 6: report `real_sup` (and every arm) beside zero flow on `indoor_flying1` dt = 1 (D-046); report the dt4 cost of dt1 selection | Claude (Phase 6) |
| D-049: floor/mask low-count pixels in the average-timestamp loss and log the clipped-step fraction before Phase 7 `maritime_selfsup` | Claude (Phase 7) |
| D-050: wire `tau_fraction_of_window` into the input build (keeping 1/3 for the trained models) | Claude (Phase 8 prep, export must match training) |
| Point `run_eval.py --max-windows` smoke runs at a tune/fixture unit, not a test sequence (auditor finding 6) | Claude |
| Store `n_events` in run provenance; hash `.v2e.json` sidecars | Claude (low) |
| Remove the job worktrees `../evflow-wt-phase5`, `../evflow-wt-phase5b` after the push | Claude |
