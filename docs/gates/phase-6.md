# Phase 6 gate — sim-to-real headline (T1/T2)

Date: 2026-10-12 · Gate run by: Claude (`/phase-gate 6`) · Code commits: 8e5aab9 (Phase 6 code, job
prepared, D-051), b19e609 (job relaunch commit: 4 CPU threads, `epe_map` as the reference formula,
D-052), 88cd806 (gate fixes: seed-pairing ranges, seed-unstable closure, D-053; CI step) (+ this
gate commit, which adds the results JSON, `results/TABLES.md` and the EPE figures)

> **Exit gate:** Tables A–D complete or each empty cell explained; every row tagged with tier and
> arm.

Plan steps (IMPLEMENTATION_PLAN.md Phase 6): (1) `scripts/run_eval.py` → per-unit summary JSON +
per-window npz; (2) `scripts/make_tables.py` → `results/TABLES.md` (Table A T1/T2 arm × sequence × dt
with CIs and 3-seed spread; Table B gap and closure with paired-bootstrap CIs, self-supervised beside
supervised; Table C representation ablation; Table D DSEC, R1 `reference_seen_data`); (3) figures
(per-sequence EPE with CIs; flow colour-wheel panels for 6 fixed MVSEC windows); (4)
`sim2real-study-runner` builds the tables, `eval-reproducibility-auditor` traces each number.

## Overall verdict: **PASS**

CI: pending on the gate push (recorded in a follow-up commit, as for Phases 0–5).

All 298 planned evaluation units are present (`results/eval/summary.json`: `n_present` 298 of
`n_expected` 298, `problems` [], `repro_check.pass` true). Every table row carries tier and arm; T1 and
T2 rows sit in separate sub-sections; there is no T3 row. Every cell absent from Tables A–D is either
omitted by design (D-051(a), stated under the table) or an undefined closure under the D-051(c)
rule (footnoted). Both required agents returned non-blocking verdicts (PASS, REPRODUCED); the
flow-metric-verifier passed the one change in `src/evflow/flow/` (D-052).

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Step 1: per-unit summary JSON (committed) + per-window npz (gitignored) for every model × unit of the matrix | PASS | `src/evflow/eval/study.py` matrix (D-051): T1 MVSEC 17 runs × 8 + 24 baseline units, T1 DSEC 11 voxel5 runs × 3 + 9, T2 MVSEC 11 voxel5 runs × 8 + M0a × 8 = 298; `results/eval/summary.json` 298/298, 0 missing; npz SHA-256 recorded in each unit and verified by the summariser; job log `runs/eval-all/job.log` (b19e609, dirty=0, no FAILED unit) |
| 2 | Step 2: `results/TABLES.md` Tables A–D rendered by script from committed JSON | PASS | `scripts/make_tables.py`; `make reproduce-tables` → `results/TABLES.md is current` (local, and in CI from this phase: `.github/workflows/ci.yml`) |
| 3 | Tables A–D complete or each empty cell explained | PASS | Table A-T1/A-T2, B-T1/B-T2 (+ T2 − T1), C, D all populated; by-design omissions (T2 for M0b/R1/ablation; DSEC for ablation runs) stated under the tables and in D-051(a); undefined closure cells footnoted with the D-051(c) rule; sim2real-study-runner completeness check PASS |
| 4 | Every row tagged with tier and arm; no T3 row in a T1/T2 section (checklist E2) | PASS | tier and arm columns in every row; renderer refuses T3 and untagged records (`tests/test_make_tables.py::test_t3_record_is_refused`, `::test_untagged_record_is_refused`); R1 tags `reference_external` (MVSEC) and `reference_external, reference_seen_data` (DSEC) |
| 5 | CIs on every T1/T2 mean; moving blocks of 1 s; paired bootstrap for arm differences (checklist C6) | PASS | `src/evflow/eval/bootstrap.py` (`moving_block_ci`, `paired_block_ci`, `stratified_paired_ci`; `tests/test_eval_bootstrap.py`); settings `configs/evflow.yaml metrics.bootstrap` recorded in every summary; Table B CIs are paired, between-seed spread shown as `r` (D-053) |
| 6 | Gap and closure as defined in plan §1.6, with paired-bootstrap CIs | PASS | `results/eval/paired.json` Table B entries; closure undefined when the CI of gap(v2e_fixed_sup) reaches 0 (D-051(c)); seed-unstable marking (D-053) |
| 7 | 3-seed mean ± range for the core arms; n_windows shown | PASS | `results/eval/aggregate.json` (`metrics`, `seed_range`, `per_seed`, `seeds_complete`); n per column in every table header |
| 8 | DSEC rows state the resolution change; R1 `reference_seen_data` | PASS | Table D header; `protocol.resolution_note` in all 33 M1 DSEC JSONs |
| 9 | Units: px over the window, forward; dt = 1 and dt = 4 never mixed in one mean (checklist C3) | PASS | every summary carries `dt`; aggregates and pooled rows are per dt (`study.DTS` loop; pooled = mean over sequences at one dt) |
| 10 | Step 3: figures | PASS | `results/figures/epe_mvsec_{T1,T2}.png` (numbers from the same JSON; committed); flow panels for the 6 windows of `configs/evflow.yaml figures` in `results/figures/flow_panels/` (MVSEC CC BY-SA imagery, gitignored, D-033/D-051(d)) |
| 11 | Results JSON carry the `event-conventions` keys; clean trees; checkpoint SHA in every eval JSON (D5) | PASS | all 298 units: tier, arm, model, seed, checkpoint_sha256, dataset, sequence, dt/window_ms, representation, metrics, ci95, n_windows, git_sha, created_by, created_at; `git_dirty` false, none `smoke`; checkpoint SHA = `selection.json` = `best.pt` for all 257 M1 units (auditor); summaries regenerated at clean 88cd806 (`git_dirty` false); `tests/test_results_have_tiers.py` passes |
| 12 | Determinism / reproduction | PASS | job reproduction unit (real_sup s1337 outdoor_day1 dt1 re-inferred at b19e609 vs its f1967de JSON): max abs diff 0.0 (`results/eval/summary.json repro_check`); auditor recomputed 3 units from npz (diff 0.0) and rebuilt `aggregate.json`/`paired.json` (diff 0) |
| 13 | No test-split use for any choice (CLAUDE.md rule 4) | PASS | eval reads frozen `best.pt` selected on `mvsec_tune` (`run_eval.load_run` refuses otherwise); nothing in Phase 6 is chosen on test data (D-051, D-053: rendering/summary rules only, panel windows fixed by position before predictions were viewed) |
| 14 | `make lint`, `make test`, `make test-data`, `make reproduce-tables` | PASS | see Commands |

## Commands (tails, at 88cd806 + gate working tree)

```
make lint            → All checks passed!
make test            → 331 passed, 26 deselected
make test-data       → 26 passed, 331 deselected
make reproduce-tables→ results/TABLES.md is current
```

## Agent verdicts

**sim2real-study-runner** — `VERDICT: PASS`. Completeness 298/298 (160 T1 MVSEC, 42 T1 DSEC, 96 T2
MVSEC); tier hygiene and required keys clean; statistics checked, pooled T1 gap recomputed from the npz
equals `paired.json` to 1e-15. Findings (none blocking), all addressed before the gate: (1) LOW Table B
showed no n → n per column added; (2) LOW repeat the D-046 caveat under Table B → reading note added
(wording refined per auditor F2); (3) INFO closure CIs wide near-zero denominators → reading note "values
above 1 are within that uncertainty"; (4) INFO T2 − T1 rows are cross-tier → stated in the sub-section
text; (5)–(7) INFO (summary tier n/a, pending changes, figures show no seed ranges — seed ranges are in
the tables).

**eval-reproducibility-auditor** — `VERDICT: REPRODUCED` (first audit and re-check of the fixes).
Regeneration current; lineage for 16 sampled + headline cells and the full chain for all 257 M1 units /
17 runs (checkpoint, manifests, config SHAs via `git show`, split and corpus SHAs, the 7 T2 stores, 298
npz SHAs); determinism diff 0.0; required keys present; claim-trace clean. Findings: **F1 Major** —
Table B CIs omit between-seed variation and the gap sign is not seed-stable (pooled T1 dt = 4
gap(v2e_fixed_sup) per seed +0.147 / −0.049 / +0.234) → fixed by D-053 (seed-pairing ranges `r`,
CI-scope sentence, seed-unstable closure marker, 4-decimal near-zero bounds), verified by the
re-check; F2/F3/F4 Low → fixed (indoor reading note names indoor_flying1 and indoor_flying3; STATUS row;
paired series carry `source_json` and checkpoint SHA); F9 Medium (re-check: summaries written from an
uncommitted summariser, `git_dirty` true) → fixed: code committed as 88cd806, summaries and tables
regenerated there (`git_dirty` false); F10/F11 Info (range format, wording) → fixed; F5–F8 Info → open
items below.

**flow-metric-verifier** (change in `src/evflow/flow/metrics.py`, D-052) — `VERDICT: PASS`. Definitions
unchanged and equal to `metrics_ref`; float64 promotion kept; Phase 4 zero-flow windows re-scored end
to end: EPE max relative difference 3.8e-16, count-based metrics bit-identical. Finding (Info):
`train/tune_eval.py` still uses `vector_norm` for the GT magnitude (equivalent to 1e-16).

## Reading of the results (for Phase 10 prose; every number below is in `results/eval/paired.json`)

- T1, pooled over the four MVSEC test sequences, 3-seed means: gap(v2e_fixed_sup − real_sup) is
  −0.019 px [−0.039, −0.0004] at dt = 1 and +0.111 px [0.074, 0.150] at dt = 4; over single-seed
  pairings it spans 0 at both dt (`seed_pairings.range`), so **its sign is not stable across training
  seeds** (D-053).
- CT randomisation: pooled T1 dt = 4 closure 0.645 [0.389, 1.057], **seed-unstable** (gap(v2e_fixed_sup)
  ≤ 0 in 2 of 9 seed pairings); undefined at dt = 1.
- Self-supervised arms (one seed each): v2e_selfsup has lower EPE than real_selfsup, pooled T1 gap
  −0.160 px [−0.197, −0.128] at dt = 1 and −0.263 px [−0.335, −0.208] at dt = 4.
- Caveat kept beside any headline: no supervised arm beats zero flow on indoor_flying1 dt = 1 (D-046),
  and every M1 arm trains on 240 s of outdoor driving (D-026, D-043).

## Incidents during the phase (all recorded in DECISIONS)

- D-052: eval-all attempt 1 ran at ≈ 5 windows/s (torch's default 18 CPU threads on WSL2; slow
  `vector_norm`); stopped after 8 min with nothing written to results/; relaunched with 4 threads and
  the reference magnitude formula; the reproduction unit shows no change (diff 0.0).
- D-053: Table B seed spread and seed-unstable closure (auditor F1).

## Open items

| Item | Who |
|---|---|
| Record the CI run of the gate push (follow-up commit) | Claude |
| F5 (Info): training manifests' `data.stores.*.manifest_sha256` names the unit-JSON SHA, and T1 units record the real store's filename but not its SHA; rename/add `store_sha256` in later phases | Claude (Phase 7+) |
| F7 (Info): D-052 timing numbers are operational and cite no log beyond `runs/eval-all/job.attempt1.log` | — (accepted) |
| Flow-metric-verifier Info: use the reference magnitude in `train/tune_eval.py` before the next training (Phase 7) | Claude (Phase 7) |
| D-049 low-count floor in the average-timestamp loss before `maritime_selfsup` (carried from Phase 5) | Claude (Phase 7) |
| Checklist C4 (eval protocol quoted in README) and E1/E3 (README claims) are README items | Claude (Phase 10) |
| Publication of the flow panels (MVSEC CC BY-SA imagery) | user (Phase 10, D-008) |
