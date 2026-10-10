# Phase 3 gate — matched-content MVSEC corpus + simulator-fidelity report

Date: 2026-10-09 · Gate run by: Claude (`/phase-gate 3`) · Code commits: 97210ab (first corpus),
a8a0f16, c737c09, a20474c, 547bd34 (first gate: FAIL), 81d1cdd (D-035 re-calibration), b6853e6
(D-035 profile frozen, corpus re-run), 02b975f, 944d870 (+ this gate commit)

> **Exit gate:** corpus manifests complete; cross-correlation median above the threshold set in
> Phase 2 on the tune slice; `v2e-fidelity-auditor` **PASS** (provenance, no test-driven choices,
> claims about fidelity limited to measured statistics).

## Overall verdict: **PASS** (second gate run, after D-035)

History: the first gate run (547bd34) was **FAIL**. The v2e-fidelity-auditor found that every unit of
the first corpus lagged the real events by ≈ 0.5 APS frame (60 Hz photoreceptor lowpass of the
Phase 2 profile). The user chose D-035: re-calibrate on `mvsec_tune` with a timing constraint,
re-run the corpus, budget limit 12 → 19 GPU-h. The threshold in the gate text ("set in Phase 2 on
the tune slice") was re-derived on the tune slice by the same rule (D-029) for the new profile; it
was not loosened, and no test data entered the choice.

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `make lint` green | PASS | `ruff check src tests scripts` → `All checks passed!` |
| 2 | `make test` green (CPU, no data) | PASS | `232 passed, 8 deselected` |
| 3 | `make test-data` green (phase touched `sim/`) | PASS | `8 passed, 232 deselected` (incl. `test_corpus.py::test_real_plan_matches_d026`) |
| 4 | Plan step 1: v2e(APS) for train/tune/test with `mvsec_cal`; K = 4 randomised train corpora, CT per 60 s clip, seeded | PASS | `results/v2e_mvsec_corpus.json`: 30/30 units in `data/converted/v2e/mvsec_d035/` (2 trials, tune, 7 test chunks, 4 train_cal clips, 16 rand), all at b6853e6 with `git_dirty` false, 0 undersampling warnings, max clamped fraction 3.19e-5; plan D-032, profile D-035; jobs `jobs/2026-10-09_v2e-mvsec-d035.{sh,manifest.json}` |
| 5 | Corpus manifests complete | PASS | every unit has `<unit>.{h5,json,v2e.json,log}`; store, run-manifest and input SHA-256 match the results JSON on all 30 units (both auditors); full A1 invariants on all 30 stores (event-data-auditor) |
| 6 | Profile chosen on tune only, timing-correct (D-035) | PASS | `results/v2e_calibration_timing.json`: 16 new tune cells + the 32 Phase 2 cells, each with an NCC-vs-lag profile; eligible = lag peak within ±0.125 frame (rule committed before the job); argmin t10 (thres 0.6, σ 0.04, cutoff 0, shot 0) reproduces from stored histograms and lag peaks (`recompute_timing_argmin`, both auditors); frozen with its SHA (`scripts/freeze_profile.py --check` → `profile frozen and consistent`) |
| 7 | Plan step 2: fidelity report per test sequence (B5) | PASS | `results/fidelity/{test_indoor_flying1..3,test_outdoor_day1,tune_outdoor_day2,train_cal,summary}.json`; W1 recomputed from the saved histograms; outdoor_day1 histograms rebuilt from raw stores by the auditor (exact); reported only (D-033) |
| 8 | Plan step 3 / exit gate: NCC median above the tune-derived threshold, per group (B4) | PASS | `results/v2e_alignment.json` `gate_pass` true: tune, indoor_flying1/2/3, outdoor_day1, train_cal all above 0.491063 (D-034); indoor_flying2 by the smallest margin, and its unit `test_indoor_flying2_p0` alone is below (stated in LIMITATIONS); 5-frame and mirror controls below the threshold in every group |
| 9 | Timing: no systematic v2e lag | PASS | per-group `lag` profile: peak +0.125 frame (one grid step) on tune, train and the three indoor sequences; outdoor_day1 (test, reported only) +0.375 with a flat profile; first corpus: +0.5 everywhere (`results/v2e_mvsec_corpus_d032_lagged.json`, gate run 1) |
| 10 | No test-driven choice (A7) | PASS | calibration and lag profiles read `mvsec_tune` only (windows checked inside tune); nothing in `src/ scripts/ configs/ jobs/` reads `results/fidelity/*` or `v2e_alignment.json`; runner split guard refuses out-of-split APS ranges |
| 11 | Fidelity claims limited to measured statistics (rule 2, `claim-trace`) | PASS | `docs/LIMITATIONS.md` Phase 3 bullets (timing/statistics trade-off, fidelity transfer, alignment scope): every number traced to a results JSON field; T2 tags; pyramid caveat; no HDR / latency / "realistic" wording |
| 12 | Trials before scaling (v2e-convert skill) | PASS | first corpus: `trial_rand_lo/hi` inspected before launch (no warnings; rates 1.45 M / 0.59 M vs 1.17 M ev/s real and NCC 0.66 / 0.60 from the session inspection). D-035 corpus: `trial_rand_lo` (0.40, σ 0.052) / `trial_rand_hi` (0.78, σ 0.02): no undersampling, `too_many_events` 0, clamped 0, `git_dirty` false; 2.00 M / 0.71 M ev/s vs 1.17 M real; NCC 0.71 / 0.68 > 0.491; lag peak 0 / +0.125; 1-frame control 0.53 / 0.51 (NCC and lag values from the session inspection before the launch at 10:59, not a results file; rates in `results/v2e_mvsec_corpus.json`) |
| 13 | Splits re-verified from a cold hash cache (A6) | PASS | `python scripts/make_splits.py --rehash` → `splits OK`; repeated by the event-data-auditor (cache redirected) |
| 14 | Budget re-costed | PASS | `results/v2e_throughput.json` `scenarios.chosen` within the new 19 GPU-h limit (user, D-035): Phase 2 measured lines + first corpus + timing calibration + D-035 corpus + the 78 unit-minutes lost at the pause (D-036). Auditor note: job wall spans exceed summed unit wall / 4 because of single-unit tails, so Phase 7 headroom is smaller than the figure suggests; re-cost before Phase 7 |
| 15 | Artefacts committed; results JSON keys | PASS | `git ls-files` lists the results above, `configs/v2e_corpus_mvsec.yaml`, `configs/v2e_profiles.yaml`, all four Phase 3 jobs; `tests/test_results_have_tiers.py` green (metric-free files: `kind` + tier, D-036) |
| 16 | CI green on the pushed commit | PASS | GitHub Actions `ci` run 38011834737 on bd2f89f: `ruff check` → `All checks passed!`, `pytest -q` → `232 passed, 8 deselected` |

Checklist lines closed (VALIDATION_CHECKLIST.md): B4, B5. A7 evidence for Phase 3 recorded (line stays
open until Phase 5 selection).

## Agents

First gate run (a20474c):

**v2e-fidelity-auditor** → `VERDICT: FAIL`. Finding 1 (HIGH): ≈ 0.5-frame v2e lag in the whole corpus
(cutoff_hz 60), undetected by the threshold. Findings 2–6 (scope text, trial record, span-based GPU
time, STATUS, docstring), all fixed.

**event-data-auditor** → `VERDICT: PASS`. A constant clock shift judged not a legitimate fix
(intensity-dependent delay), which led to D-035's simulator-setting fix.

Second gate run (02b975f; both started together, read-only and independent — the skill asks for one
at a time):

**v2e-fidelity-auditor**:

> VERDICT: PASS

Choice reproduces on tune only (objectives to 1e-9, lag peaks, near cells, threshold 0.75 × 0.65475);
all 30 manifests and draws verified; kept units unaffected by the pause; eligibility rule a sound
coarse filter (residual lag below 1/8 frame by its finer check). Lows, all addressed in 944d870:
threshold-source string (regenerated), pyramid bullet wording, outdoor_day1 lag numbers and tune/train
hot-pixel W1 in LIMITATIONS, per-unit indoor_flying2_p0 below threshold disclosed, span-based budget
note (criterion 14), gate/STATUS refresh (this file).

**event-data-auditor**:

> VERDICT: PASS

A1 invariants with full `ms_to_idx` on all 30 stores (1.05 B events), provenance hashes, clock map
exact at the knots, no output mixes the killed and resumed attempts, split leakage clean, D-026 clips,
window ownership, argmin and SHA reproduce. Minors, all addressed in 944d870: threshold source, pause
GPU time (budget line, D-036), docstring, tier "n/a" of metric-free files (D-036).

## Open items

| Item | Who |
|---|---|
| Phase 5 loaders: assert the unit manifest's split and `run_manifest_sha256`; skip the first 10 windows of every train/tune clip, all arms (D-032c) | Claude, Phase 5 |
| Re-cost the v2e budget (span-based) before the Phase 7 maritime conversions; the maritime profiles inherit the D-035 pixel model (cutoff 0) — trial `maritime_640x360` (5 s) first | Claude, Phase 7 |
| First (lagged) corpus `data/converted/v2e/mvsec/` (3.1 GB, gitignored) is superseded; kept for now, can be deleted | user |
| Stray v2e output folder from Phase 2 under `external/v2e/data/` (carried over) | user |
