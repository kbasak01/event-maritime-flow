# Phase 3 gate — matched-content MVSEC corpus + simulator-fidelity report

Date: 2026-10-09 · Gate run by: Claude (`/phase-gate 3`) · Code commits: 97210ab (corpus job),
a8a0f16, c737c09, a20474c (+ this gate commit)

> **Exit gate:** corpus manifests complete; cross-correlation median above the threshold set in
> Phase 2 on the tune slice; `v2e-fidelity-auditor` **PASS** (provenance, no test-driven choices,
> claims about fidelity limited to measured statistics).

## Overall verdict: **FAIL** (v2e-fidelity-auditor FAIL: systematic ≈ 0.5-frame v2e timing lag)

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `make lint` green | PASS | `ruff check src tests scripts` → `All checks passed!` (a20474c) |
| 2 | `make test` green (CPU, no data) | PASS | `226 passed, 8 deselected` |
| 3 | `make test-data` green (phase touched `sim/`) | PASS | `8 passed, 226 deselected` (incl. `test_corpus.py::test_real_plan_matches_d026`) |
| 4 | Plan step 1: v2e(APS) for train/tune/test with `mvsec_cal`; K = 4 randomised train corpora, CT per 60 s clip, seeded | PASS | `results/v2e_mvsec_corpus.json`: 30/30 units (2 trials, tune, 7 test, 4 train_cal, 16 rand), `all_git_clean` true (97210ab), 0 undersampling warnings, max clamped fraction 3.05e-5; D-032; job `jobs/2026-10-09_v2e-mvsec.{sh,manifest.json}` |
| 5 | Corpus manifests complete | PASS | every unit has `<unit>.{h5,json,v2e.json,log}`; store / run-manifest SHA-256 match the results JSON (both auditors, all 30 units) |
| 6 | Plan step 2: fidelity report per test sequence (B5) | PASS | `results/fidelity/{test_indoor_flying1..3,test_outdoor_day1,tune_outdoor_day2,train_cal,summary}.json`; W1 recomputed from saved histograms (auditor, 1e-14); reported only (D-033) |
| 7 | Plan step 3 / exit gate: NCC median above the Phase 2 tune threshold, per sequence (B4) | PASS (with caveat) | `results/v2e_alignment.json`: `gate_pass` true for tune, 4 test sequences, train_cal (threshold 0.4407 from `configs/v2e_profiles.yaml`, unchanged since 79bddba). Caveat: the threshold and the 1-frame control cannot see the lag of criterion 8 |
| 8 | Alignment is correct, i.e. the check's purpose ("a frame-timing … bug before any training") | **FAIL** | v2e events lag real events by ≈ 0.5 APS frame (≈ 11 ms on outdoor_day2) in every `mvsec_cal` and `mvsec_rand` unit. Cause verified on tune calibration units only: cells with cutoff_hz 0 (c20, c28) peak at lag 0; cells with cutoff_hz 60 (c22, c30 = `mvsec_cal`) peak at +0.5 frame. v2e's photoreceptor lowpass delays events, more for darker pixels. The calibration objective (W1 of distributions) has no timing term |
| 9 | No test-driven choice (A7) | PASS | no code/config reads `results/fidelity/*` or `v2e_alignment.json`; calibration reads `mvsec_tune` only; the runner split guard refuses out-of-split APS ranges (event-data-auditor probes) |
| 10 | Fidelity claims limited to measured statistics (rule 2, `claim-trace`) | PASS | `docs/LIMITATIONS.md` Phase 3 bullets: every number traced to a results JSON field (claim-trace CLEAN after two fixes: rate ratio now written by the script, T2 tag added); IEI pyramid caveat in every fidelity JSON |
| 11 | Splits re-verified from a cold hash cache (Phase 2 open item, A6) | PASS | `python scripts/make_splits.py --rehash` → `splits OK` (3 min 35 s; repeated independently by the event-data-auditor) |
| 12 | Trials before scaling (v2e-convert skill) | PASS (late record) | `trial_rand_lo` / `trial_rand_hi` (5 s at thres 0.28/σ 0.052 and 0.52/σ 0.02): no undersampling, `too_many_events` 0, clamped 0, `git_dirty` false; rates 1.45 M and 0.59 M ev/s vs 1.17 M real; NCC 0.66 / 0.60 (> threshold), shift-5 0.35 / 0.34, mirror 0.15 / 0.14. Inspected in the session before launch (24 s gap), recorded here at the gate's request |
| 13 | Budget re-costed after the corpus | PASS | `results/v2e_throughput.json` `scenarios.chosen` 11.08 GPU-h (Phase 3 lines measured: 5.51 + 0.08). Auditor note: the job's wall span was 5.67 h (single-unit tail), so the span-based total is ≈ 11.24 GPU-h; Phase 7 headroom ≈ 0.76 GPU-h |
| 14 | Artefacts committed; results JSON keys | PASS | `git ls-files` lists the results above, `configs/v2e_corpus_mvsec.yaml`, the job; `tests/test_results_have_tiers.py` green (`kind` + `tier T2`) |

Checklist lines (VALIDATION_CHECKLIST.md): B5 met; B4 reported per sequence with the threshold fixed
before any test conversion, but **not ticked** until criterion 8 is resolved.

## Agents

**v2e-fidelity-auditor** (a20474c):

> VERDICT: FAIL

1. HIGH — ≈ 0.5-frame v2e lag in the whole corpus, not detected by the gate and not reported
   (NCC-vs-lag: tune peak at +0.5 frame, 0.617 vs 0.579 at 0; train_clip0, trial_rand_hi likewise;
   cause on tune: cutoff_hz 0 cells peak at 0). Fix: a D-entry decided on `mvsec_tune` only (re-run
   with a different lowpass, timing term in the profile choice, or a documented correction /
   acceptance), and an NCC-vs-lag profile in `alignment_check.py`.
2. MEDIUM — `scope` text of the alignment JSON claimed 1-frame timing errors are detected; LIMITATIONS
   omitted that shift1 exceeds aligned in 7/10 groups; the standstill sentence was untraced.
3. LOW — trial inspection not recorded (criterion 12). 4. LOW — span-based GPU time (criterion 13).
5. LOW — STATUS still listed the job as pending. 6. INFO — "no signal events" v2e log lines are
   benign; `corpus.py` docstring flag.

Everything else re-run independently and passed: argv policy and provenance on all 30 units, draws,
window ownership, identical treatment of real and v2e (12 histograms recomputed from raw stores),
W1 recomputation, threshold provenance, no test-driven choice, LIMITATIONS numbers.

**event-data-auditor** (a20474c):

> VERDICT: PASS

Store invariants on all 33 new stores (3 real, 30 v2e; full `ms_to_idx` recomputation), converter
fidelity vs source, provenance hashes, clock map exact at the knots, clamped ≤ 3.05e-5, split guard
probes all refused, D-026 clips exact, every test window owned once, A7 grep clean, `--rehash` →
`splits OK`. Minors: (1) record the rehash (criterion 11); (2) `corpus.py` docstring flag; (3) Phase 5
loaders must assert the unit manifest's split and skip the first 10 windows (D-032c). Comment on the
lag: a constant clock shift is **not** a legitimate fix — the clock map is correct, the lag is the
simulated pixel's intensity-dependent lowpass response, and a shift fitted on bright outdoor_day2
would under-correct darker scenes; the proper fix is a simulator setting re-chosen on `mvsec_tune`.

## Fixed after the audits (not yet re-run)

- `scripts/alignment_check.py`: corrected `scope`; new per-group `lag` profile (median NCC with the
  v2e window delayed by −0.5 … +1 APS interval, `peak_lag_frames`), `sim/fidelity.py`
  `unit_lag_ncc` / `lag_summary` with a test; `corpus.py` docstring.

## Open items

| Item | Who |
|---|---|
| Decide the timing fix (D-035), on `mvsec_tune` evidence only, and the GPU-budget consequence | **user** |
| Implement the decision; re-run the affected corpus, fidelity report and alignment check (with lag profile); re-derive the NCC threshold on tune if the profile changes; re-run both auditors | Claude |
| Rewrite the LIMITATIONS alignment bullet from the lag-profile JSON; drop or trace the standstill sentence | Claude |
| Phase 5 loaders: assert unit-manifest split; skip first 10 windows of every train/tune clip (D-032c) | Claude, Phase 5 |
| Phase 7 maritime trial (`maritime_640x360`, 5 s) before the maritime corpus | Claude, Phase 7 |
