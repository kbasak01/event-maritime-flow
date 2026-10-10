# Phase 4 gate — baselines and reference reproduction gate

Date: 2026-10-10 · Gate run by: Claude (`/phase-gate 4`) · Code commits: 0da94b3 (Phase 4 code, job
prepared; every baseline unit, the M0b selection and the reproduction were produced at this commit,
`git_dirty = false`), 3663684 (R1 diagnostic script), b9346c5 (diagnostic fix after the verifier's
finding 1; `r1_mvsec_diagnostic.json` regenerated at it) (+ this gate commit)

> **Exit gate:** `results/baselines/*.json`; reproduction gate met or the miss explained in
> `docs/DECISIONS.md` and accepted by `flow-metric-verifier`.

Plan steps (IMPLEMENTATION_PLAN.md Phase 4): (1) M0a, M0b on `mvsec_test` (dt = 1, 4) and
`dsec_test`; (2) R1 (E-RAFT DSEC checkpoint) on `dsec_test`, tagged `reference_seen_data`; (3) the
reproduction gate on E-RAFT's preprocessed `outdoor_day1` 20 Hz release: our metric code reproduces
E-RAFT's own evaluation on the same flows to 1e-3 px, and the EPE is within 10 % of the paper.
The user additionally asked for R1 on `mvsec_test` (D-040).

## Overall verdict: **PASS**

Every Phase 4 result is tier **T1** (real events, real GT); MVSEC units use the windows and masks
of `configs/eval_mvsec.yaml` (D-019), the reproduction gate E-RAFT's 20 Hz windows
(`reproduction_eraft`, D-039).

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | M0a (zero flow) on `mvsec_test` dt 1 and 4 and `dsec_test` | PASS | `results/baselines/zero_flow/` (11 files); FWL exactly 1 in every window (verifier item 3) |
| 2 | M0b (patch-wise CMax, own implementation) on the same units, parameters chosen on `mvsec_tune` only | PASS | `results/baselines/cm_baseline/` (11 files); selection `results/baselines/cm_tune.json` (split `mvsec_tune`, argmin tile 64 px / TV 0.01, read by `scripts/run_baselines.py`, D-038) |
| 3 | R1 on `dsec_test` (DSEC checkpoint, 15-bin voxels, 100 ms), tagged `reference_seen_data` | PASS | `results/baselines/eraft_dsec/` (3 files, tags `reference_external`, `reference_seen_data`) |
| 3b | R1 on `mvsec_test` (user request) | PASS (with caveat) | `results/baselines/eraft_mvsec20/` (8 files, `reference_external`); worse than zero flow at indoor dt = 1, explained as a generalisation gap (D-042, `results/baselines/r1_mvsec_diagnostic.json`) |
| 4 | Reproduction gate: metric parity ≤ 1e-3 px with E-RAFT's own evaluation on the same flows | PASS | `results/baselines/eraft_reproduction.json` `parity`: max per-sample \|ΔEPE\| 1.38e-9 px dense, 2.52e-9 px sparse; masks identical to E-RAFT's (0 differing pixels); `gate.metric_parity_1e-3px = true`. Definition of "E-RAFT's own evaluation": D-039 (user decision) |
| 5 | Reproduction gate: EPE within 10 % of the paper | PASS | same file: E-RAFT dense EPE 0.4674 vs 0.47 (arXiv:2108.10552v2 Table 3), `paper.rel_diff_dense_epe` −0.56 %; `gate.epe_within_10pct_of_paper = true` |
| 6 | Every results JSON carries the required keys (`event-conventions`) and comes from a clean tree | PASS | 36 metric files checked for tier, arm, model, seed, checkpoint_sha256, dataset, sequence, dt/window_ms, representation, metrics, ci95, n_windows, git_sha, created_by, created_at; none `smoke`, none `git_dirty`; `summary.json` (kind `baselines_index`): 33/33 units, no problems |
| 7 | Artefacts committed | PASS | `git ls-files results/baselines` lists all 37 JSON files (this gate commit) |
| 8 | `make lint` | PASS | `All checks passed!` |
| 9 | `make test` | PASS | `257 passed, 22 deselected, 4 warnings in 18.70s` |
| 10 | `make test-data` (phase touched `eval/`) | PASS | `22 passed, 257 deselected in 8.41s` |
| 11 | No test split used for any choice | PASS | M0b grid on `mvsec_tune` (`cm_tune.json` `split`); the grid revision (D-038) used a tune-slice smoke run; R1 and M0a have no parameters chosen; diagnostics on test sequences are reported only (D-042) |
| 12 | Licences: vendored E-RAFT keeps MIT header, RAFT BSD-3 notice; no GPL | PASS | `src/evflow/models/eraft_ref/` headers + `LICENSE`; `docs/LICENSES.md` rows E-RAFT (vendored), princeton-vl/RAFT; `tests/test_no_gpl_imports.py` passes |

Checklist: **C5** met (reproduction gate) → ticked in `VALIDATION_CHECKLIST.md`.

## Agent verdicts

**flow-metric-verifier** (first run, on the results and code at 0da94b3/3663684):

> VERDICT: PASS

Findings:
1. (medium) The "warm_start" arm of `r1_mvsec_diagnostic.json` was in fact cold (isolated picks
   never had their predecessor run), so D-042's statement about warm start was unsupported; the
   conclusion (not a pipeline error) survived. → Fixed in b9346c5: 10 unscored predecessors per
   warm pick; diagnostic regenerated; D-042 rewritten.
2. (low) D-042 not qualified by dt (R1 beats zero at indoor dt = 4). → D-042 now says dt = 1.
3. (low) Supporting evidence for D-042: R1/zero > 1 in every motion band at indoor dt = 1; M0b
   beats zero on the same units. → Added to the diagnostic JSON (`by_motion_band`) and to D-042.
4. (info) Provenance: results from 0da94b3, `git_dirty = false`; results and D-042 were not yet
   committed at review time. → Committed with this gate.

Independent checks the verifier reported: hand-computed EPE / %Out / NPE / AE cases; reproduction
rebuilt with numpy + h5py only (per-sample EPEs and means identical, event-file/window pairing and
masks verified); 10 MVSEC and 4 DSEC zero-flow windows recomputed from the raw files, identical.

**flow-metric-verifier** (re-check of findings 1–2 after b9346c5):

> VERDICT: PASS

Findings 1 and 2 resolved (verifier's own recomputation from the npz files): the warm arm now matches the units' mean EPE on the same 200 picks to
within 1.4e-3 px (10 predecessors, not the full chain) and differs from cold; D-042's dt = 1
qualifier and claims (a)–(c) are supported by `r1_mvsec_diagnostic.json`. Note 3 (low,
documentation): claim (e) cites numbers in `results/baselines/cm_baseline/` and `zero_flow/`, not in
the files D-042 listed. → Evidence column of D-042 extended. Note 4 (info): commit results and
D-042 with the gate. → Done in this gate commit.

## Findings recorded for later phases

- R1 (20 Hz MVSEC checkpoint) is worse than zero flow on indoor_flying at dt = 1 (D-042). R1 is a
  reference row only; Phase 6 tables carry the caveat.
- M0b selected on `mvsec_tune` does not beat zero flow on the tune slice (`cm_tune.json`
  `zero_flow_reference`); it does on the indoor units, at dt = 4 and on DSEC. Kept as selected.
- GPU time: M0b took 3.7–5.3 s per window with six processes sharing the A4000
  (`mean_pred_s_per_window` in `results/baselines/cm_baseline/*.json`), so the job's wall time
  (start/end timestamps in `runs/phase4-baselines/job.log`) exceeded the 2–3 h estimate in the job
  manifest.

## Open items

None blocking. Pending GPU jobs: none. Next: Phase 5 (not started; Claude does not start it
without the user).

## CI

Gate commit e58e6d9 pushed to `main`; GitHub Actions run 38041591150 (`ci`): completed, success.
