# Phase 7 gate — Maritime track (T3) and sea-clutter study

Date: 2026-10-12 · gate commit 828b3e4 · Overall: **PASS (CI green)**.

## Gate text (IMPLEMENTATION_PLAN.md, Phase 7)

> **Exit gate:** T3 tables in their own section of `results/TABLES.md` with tier tags; clutter
> results JSON + figure; `sim2real-study-runner` confirms no T3 row sits in a T1/T2 section.

Steps 1–6 of the phase: v2e(SMD, LaRS) corpus with the calibrated profile and two CT variants;
RAFT teacher with forward–backward mask; `maritime_selfsup` and `maritime_teacher` (one seed each);
T3a/T3b evaluation of every model on `smd_test` and `lars_eval`, class-sliced on LaRS, plus the
maritime arms on `mvsec_test`; sea-clutter study with LaRS scene attributes; qualitative panels if
licences permit.

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | T3 tables in their own section with tier tags | PASS | `results/TABLES.md` part "# Maritime tables (T3)": Tables F-T3a, G-T3b, H; every row tagged; `scripts/tables_maritime.py` refuses T1/T2 records and the key `epe` there (`tests/test_make_tables.py::test_t3_part_refuses_wrong_tiers_and_epe`) |
| 2 | No T3 row in a T1/T2 section (checklist E2) | PASS | sim2real-study-runner re-check (below); `make_tables._check_tier` refuses T3 above the T3 part; Table E (maritime arms on real `mvsec_test`) is T1 and sits in the T1/T2 part |
| 3 | Clutter results JSON + figure | PASS | `results/maritime/clutter.json` (tier T3b), `results/figures/clutter_ct.png` (both tracked) |
| 4 | Step 1: maritime v2e corpus | PASS | `results/v2e_maritime_corpus.json`: 243/243 units, all git-clean at 7fd867c, 0 undersampling warnings (D-054); budget 19.44 within the user-raised 19.5 GPU-h (D-059) |
| 5 | Step 2: RAFT teacher + FB mask | PASS | `results/teacher/summary.json`: 161 units, clean at cdc358b (D-055) |
| 6 | Step 3: both maritime arms trained, no collapse | PASS (caveats) | `results/train/maritime_summary.json` 2/2 done at b552569; training-run-diagnostician TRUSTWORTHY WITH CAVEATS × 2 (D-062) |
| 7 | Step 4: T3a + T3b, class slices, maritime arms on `mvsec_test` | PASS | `results/eval_maritime/summary.json` 176/176 units (T1 16, T3a 80, T3b 80), no missing cell |
| 8 | Step 5: clutter study with LaRS attributes that exist | PASS | attributes inspected in `image_annotations.json` before use (D-058); Table H |
| 9 | Step 6: qualitative panels within licences | PASS | SMD imagery not drawn; LaRS GIFs in gitignored `results/figures/maritime_panels/`, publication left to Phase 10 (D-057f, D-008) |
| 10 | Results JSON required keys / tiers | PASS | all 176 unit JSONs carry the `docs/CONVENTIONS.md` keys, are git-clean, not smoke; no `epe` key in any T3 file (eval-reproducibility-auditor) |
| 11 | Commands | PASS | below |

## Commands (tails, at 4cd8c71)

```
$ make lint            ->  All checks passed!
$ make test            ->  374 passed, 29 deselected, 4 warnings in 56.19s
$ make test-data       ->  28 passed, 375 deselected in 37.11s
$ make reproduce-tables -> results/TABLES.md is current
```

## Agent verdicts

**sim2real-study-runner** — first run (c71bcea): `VERDICT: FAIL`. Blocking: the Table E caption
lacked the diagnostician's required caveat that the maritime arms' MVSEC rows are zero-shot,
out-of-domain, single-seed results; low: no Table G context for `maritime_selfsup` leading on
FWL, no window counts in Tables F/G, no top-level `tier` in the T3 summary. All mechanical checks
passed. Fixed in 4cd8c71 (numbers unchanged). Re-check (4cd8c71): **`VERDICT: PASS`** — "All
four findings from my last review are fixed at HEAD 4cd8c71, and I found nothing new"; Tables A–E
above `# Maritime tables (T3)`, Tables F, G, H below it; no T3 row above, no T1/T2 row below.

**eval-reproducibility-auditor** (4cd8c71): **`VERDICT: REPRODUCED`**. Table cells → summary →
unit JSON → npz → checkpoint → run manifest → corpus / teacher → split manifests verified; summary
rebuilt identically (720 entries); two units re-run on the GPU with every metric array equal to
the committed npz (max |Δ| 0); clutter.json rebuilt (max diff 0). Findings: (1) minor —
LIMITATIONS still said "SMD (45 s train, 30 s test)" → fixed to 44.8 s / 29.3 s; (2) info — the
training summary and corpus file carry no top-level `tier`/`git_dirty` (Phase 5 precedent;
`checks.all_git_clean` true); (3) info — per-window npz SHAs are not bit-reproducible because they
store the prediction wall time `pred_s`; reproduction compares metric arrays; (4) info — D-054
now names the SMD test videos; (5) info — D-062(c) wording accepted.

**Required before or within the phase** (CLAUDE.md agent roles):
- flow-metric-verifier (new teacher / T3 metric code): `VERDICT: PASS`; minor findings fixed
  (RSAT on windows defined for every model, seed-window alignment, resample counts, tests, D-057
  notes).
- event-data-auditor (readers, corpus, split use): `VERDICT: PASS`; all 243 stores checked;
  findings fixed (LaRS clips from SMD footage: no copies, D-061; text corrections).
- training-run-diagnostician: `maritime_teacher_voxel5_s1337` **TRUSTWORTHY WITH CAVEATS**,
  `maritime_selfsup_voxel5_s1337` **TRUSTWORTHY WITH CAVEATS** (verbatim in
  `docs/gates/phase-7-diagnostics/`; D-062).
- `/claim-trace` on D-054…D-062 and the Phase 7 LIMITATIONS lines: 6 findings, all fixed (one
  wrong number corrected: the pre-launch v2e estimate was 1.75 GPU-h; untraced numbers removed or
  sourced); re-checked by the eval-reproducibility-auditor.

## Provenance notes

- Corpus at 7fd867c, teacher at cdc358b, training at b552569 (clean detached worktree
  `emf-wt-v2e-maritime`); evaluation in the main checkout from 5eea7f9: 12 units record 5eea7f9
  and 164 record 3cfe6fa, two commits that differ in `docs/` only (diagnostic reports), committed
  while the job ran; no evaluated code changed (auditor finding 5).
- Training attempt 1 (4164c3d) failed at epoch 0 on an empty-tail store; fix and relaunch
  recorded in the D-060 amendment (`runs/train-maritime/attempt1/`).
- v2e budget: measured 19.44 GPU-h vs the 19 GPU-h limit; reported to the user, who accepted the
  overrun; limit set to 19.5 (D-059).

## Validation checklist

- E2 — re-checked for Phase 7: met (criterion 2).
- E5 — maritime metrics per class wherever LaRS labels exist: met (Tables F, G per class on LaRS
  keyframe windows; Table H per class).
- E7 — LaRS frame-rate assumption next to every LaRS number: met (T3 protocol paragraph, Table H
  caption, clutter figure title, D-024).
- G4 — LaRS / SMD imagery only where `docs/LICENSES.md` permits: met for this phase (no SMD
  imagery; LaRS panels gitignored, publication decision in Phase 10).

## CI

- 828b3e4 (gate commit): run 38315084301 **failed** — `tests/test_io_smd.py::test_probe_and_read`:
  `FileNotFoundError: 'ffprobe'` on the GitHub runner (373 other tests passed).
- 8b6f614 (CI installs ffmpeg so the probe stays tested): run 38315269762 **success**
  (https://github.com/kbasak01/event-maritime-flow/actions/runs/38315269762).

## Open items

| Item | Who |
|---|---|
| Publication of the LaRS qualitative GIFs (CC BY-NC 4.0, attribution in each frame) | user (Phase 10, D-008) |
| Optional: `tier: "n/a"` in `scripts/summarise_training.py` outputs (precedent accepted) | Claude (Phase 9 sweep) |
| Checklist C4 / E1 / E3 README items | Claude (Phase 10) |
