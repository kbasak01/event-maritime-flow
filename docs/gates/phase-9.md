# Phase 9 gate — validation sweep

Date: 2026-10-12 · base commit 8928598 (+ the Phase 9 docs-only commit) · Overall: **PASS**.
No code, config or `results/` file changed in Phase 9 (docs, `VALIDATION_CHECKLIST.md`,
`.gitignore` only); no GPU job.

## Gate text (IMPLEMENTATION_PLAN.md, Phase 9)

> Walk `VALIDATION_CHECKLIST.md` line by line; verdict + evidence (met / met-with-caveat / unmet) into
> `docs/LIMITATIONS.md`. Run `eval-reproducibility-auditor` and `repo-publication-reviewer` (dry run).
> Re-run `make reproduce-tables` from committed JSON and diff against `results/TABLES.md`.
>
> **Exit gate:** every line has a verdict; no critical finding open.

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Every checklist line has a verdict + evidence in `docs/LIMITATIONS.md` | PASS | 42/42 lines (A1–G6) have a row; the row IDs match the checklist IDs one to one (scripted diff); 39 met / met-with-caveat (ticked in `VALIDATION_CHECKLIST.md`), 3 unmet |
| 2 | No critical line unmet | PASS | Critical lines A1, A2, A7, C1, C2, E1, E2, E3, G1, G2: A1, A2, C1, C2, E2, E3, G1, G2 **met**; A7, E1 **met-with-caveat** (A7: D-064 scratch-run disclosure; E1: README traced in Phase 10) |
| 3 | No critical finding open | PASS | eval-reproducibility-auditor: no blocking finding (2 major fixed, below); repo-publication-reviewer dry run: "Phase 9 blockers: None" |
| 4 | `make reproduce-tables` from committed JSON, diff against `results/TABLES.md` | PASS | `--check` "results/TABLES.md is current"; re-render in a clean worktree of 8928598 (`PYTHONPATH=src python scripts/make_tables.py`): `diff` empty, 545 lines; auditor repeated it independently (byte-identical) |
| 5 | eval-reproducibility-auditor run | PASS | `VERDICT: REPRODUCED` |
| 6 | repo-publication-reviewer (dry run) run | PASS | `VERDICT: SHIP (dry run, Phase 9 only)` |
| 7 | Artefacts committed | PASS | `git ls-files docs/LIMITATIONS.md VALIDATION_CHECKLIST.md results/TABLES.md`; this file and the edits in the Phase 9 commit |
| 8 | Commands | PASS | below |

### Unmet lines

- **Critical:** none.
- **Non-critical:** C4 (eval protocol quoted in README) and F4 (A4000 stand-in sentence in README):
  README not yet written, Phase 10 (Claude). F1 (fp16 parity ≤ 5e-2 px): fp16 rows fail, accepted
  non-conformance D-064 (user decision); fp32 and EPE-change clauses met; tolerance unchanged.

## Checks run for the walk (Phase 9)

- A6: `scripts/make_splits.py --check` → `splits OK`.
- A7: all 33 `runs/**/selection.json` → `split: mvsec_tune`; `split: mvsec_tune` in
  `results/v2e_calibration.json`, `v2e_calibration_timing.json`, `baselines/cm_tune.json`,
  `train/cm_sweep.json`, `fp16_sensitivity.json`; test-sequence names in `configs/` and run
  manifests only in evaluation-only fields (v2e test corpus, `export.parity`, `figures.panel_windows`).
- D5: 403 eval JSONs of trained runs carry `checkpoint_sha256` = the run's `selection.json` (0 missing, 0 mismatched).
- E3/E4/E6: grep of `docs/`, `results/TABLES.md`, `scripts/`, `src/`, `configs/`: forbidden words only in
  disclaimers; plain "EPE" only in T1/T2 tables; no novelty claim.
- F2/F3: `results/latency.json` `methodology`, `environment.{torch_process,ort_process}`.
- G2: 778 tracked files; the only images ever added (whole history) are the three metric plots in `results/figures/`.
- G5: headers of `src/evflow/models/eraft_ref/*.py`, `tests/oracles/event_flow_metrics.py`.
- **claim-trace** (E1) on `docs/LIMITATIONS.md` (all bullets, verdict table, new D-064 bullet) and
  `docs/DECISIONS.md` D-063, D-064. All Phase 2/3/7/8 numbers matched their JSON field after rounding
  (`results/fidelity/summary.json`, `v2e_alignment.json`, `v2e_calibration{,_timing}.json`,
  `v2e_{mvsec,maritime}_corpus.json`, `onnx_parity.json`, `fp16_sensitivity.json`,
  `train/summary.json`, `latency.json`, `configs/v2e_profiles.yaml`). **2 findings, fixed**: D-063(d)
  quoted job-smoke values ("≤ 1e-4 px", "0.473 px", "≤ 0.0287 px") whose only source is the gitignored
  `runs/export-bench/smoke/onnx_parity.json`, and D-063(e) a log count ("11 per session"); both reworded
  without the numbers. Audit counts in the verdict table (393 tests, 33 selections, 403 eval JSONs,
  778 files) are command outputs recorded here, not metrics. Wording checks: clean.
  After the auditors: CLEAN.

## Agents

**eval-reproducibility-auditor**: `VERDICT: REPRODUCED`. No blocking issues; no critical checklist
line actually unmet. Verified: tables re-render byte-identical in a clean worktree; `pytest` 393
passed; 10 per-arm eval JSONs recomputed from stored npz (diff 0.0) and 6 Table B paired-bootstrap
entries re-implemented (≤ 2e-16); full checkpoint → manifest → config → split lineage for the 19
trained runs; no results JSON at a dirty commit, all `git_sha` ancestors of HEAD, all `created_by`
scripts present; D5, A7, every LIMITATIONS / D-064 number; C4/F1/F4 correctly unmet and non-critical.
It also re-checked all 8 real MVSEC/DSEC stores (t int64 non-decreasing, p ∈ {0,1}, x < width,
y < height, `ms_to_idx` = full recompute): pass.
- M-1 (`docs/gates/phase-9.md` cited but missing) → resolved by this file.
- M-2 (D-063(a) said the v2e arms logged no real-stream tune EPE; `results/train/summary.json`
  `tune_at_selected_epoch.real_dt1` has 0.293–0.334 for the supervised v2e runs) → **fixed**:
  D-063(a) reworded (selection on own stream; real-stream values cited; the choice itself was
  tune-only and is unchanged, A7 unaffected).
- m-1 (A1 evidence) → A1 row cites Phase 3/7 store checks and this 8-store check.
- m-2, m-3, m-5 (no `git_dirty` in Phase 2/3 JSON; committed lineage stops at `manifest_sha256`
  because `runs/` is gitignored; `splits/SHA256SUMS.txt` lists canonical-JSON SHAs, not
  `sha256sum -c` input) → disclosed in a LIMITATIONS "Provenance" bullet (`splits/README.md` is
  generated by `make_splits.py` and was not hand-edited).
- m-4 (E3 evidence list incomplete) → extended.

**repo-publication-reviewer (dry run)**: `VERDICT: SHIP (dry run, Phase 9 only) — nothing blocks
Phase 9; NOT the publication ship verdict (README and release are Phase 10; repo is still PRIVATE on
GitHub).` Phase 9 blockers: none. G1–G6 verdicts confirmed. Committed PNGs viewed: metric plots, no
dataset imagery. (Its note that claim-trace "was invoked without a target" is inaccurate: it ran on
LIMITATIONS, D-063 and D-064; only the not-yet-existing README remains.)
- M4 (`.gitignore` lacks `*.npy`, `*.gif`, `*.mp4`, `*.bag`) → **fixed** (patterns added; no tracked file matches).
- M5 (`docs/LICENSES.md` header date) → **fixed** ("on or after 2026-10-08 (date per row)").
- M1, M2, M3 → open items below.

## Commands (final tree)

```
$ make lint
ruff check src tests scripts
All checks passed!
$ make test
393 passed, 29 deselected, 6 warnings in 59.41s
$ make reproduce-tables
python scripts/make_tables.py --check
results/TABLES.md is current
$ python scripts/make_splits.py --check
splits OK
```

`make test-data` not run: Phase 9 touched no `io/`, `sim/` or `eval/` code (the auditor's 8-store
check covers the real stores).

## Open items

| Item | Who |
|---|---|
| README: quote `configs/eval_mvsec.yaml` protocol (C4), A4000 stand-in sentence (F4), fp32-only deployment claim (D-064), tier legend, §0(1) box, §0 positioning, MVSEC attribution beside the EPE plots (G3), claim-trace on README (E1, E3, E6, E7) | Claude (Phase 10) |
| LaRS imagery in a public README: terms forbid direct sharing of images; recommend plots only (no LaRS frame GIFs) and an explicit `docs/LICENSES.md` row (reviewer Phase 10 to-do 3) | user decision (Phase 10) |
| Checkpoint licensing: maritime models trained on LaRS (CC BY-NC) and SMD (no licence); MVSEC-trained weights and CC BY-SA (reviewer to-do 4) | user decision (Phase 10) |
| Release assets (checkpoints + ONNX + `SHA256SUMS.txt`), G2 re-check on the release; run manifests shipped or summarised (auditor m-3) | Claude (Phase 10) |
| Absolute home paths in 17 files under `docs/gates/phase-{5,7}-diagnostics/` → repo-relative before going public (reviewer M1) | Claude (Phase 10) |
| Commit author email in history (reviewer M2): fine if intended, otherwise only a history rewrite before the first public push | user |
| `src/evflow/io/mvsec.py` `_propagate` mirrors EV-FlowNet's propagation step by step (Penn non-commercial; re-implemented from the description, D-019; low risk); optional restructuring for margin (reviewer M3) | user decision |
| Phase 7 optional item (`tier: "n/a"` in training summaries) not done: it would regenerate results JSON and is not required | — |

Pending GPU jobs: none.
