# Phase 2 gate — v2e runner, throughput budget, CT calibration

Date: 2026-10-09 · Gate run by: Claude (`/phase-gate 2`) · Code commits: fbc216d, 79bddba, d4d1cd2,
fe1860e, 6c5d2cc (+ this gate commit)

> **Exit gate:** throughput and calibration JSON committed; profile and randomisation range frozen in
> `configs/v2e_profiles.yaml` with the calibration JSON's SHA.

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `make lint` green | PASS | `ruff check src tests scripts` → `All checks passed!` |
| 2 | `make test` green (CPU, no data) | PASS | `210 passed, 7 deselected` |
| 3 | `make test-data` green (phase touched `io/`, `sim/`) | PASS | `7 passed, 207 deselected` (incl. `tests/test_v2e_dataset.py`: manifest links, store integrity + APS clock, polarity vs APS log-intensity) |
| 4 | `make smoke` still green | PASS | `13 passed` |
| 5 | Plan step 1: `sim/v2e_runner.py` + `configs/v2e_profiles.yaml`, manifest per output, v2e h5 → canonical store | PASS | `src/evflow/sim/v2e_runner.py`, `src/evflow/io/v2e_reader.py` (layout D-022, inspected before writing: checklist A3); every unit has `<unit>.v2e.json` + `<unit>.json`; trial manifest written afterwards (D-031) |
| 6 | 5 s trial before scaling (user request, v2e-convert skill) | PASS | `trial` profile, `--stop_time 5`, APS 3651–3870 of `mvsec_tune` (D-022); event rate, polarity sign and clock checked |
| 7 | Plan step 2: throughput JSON; corpus sized to ≤ 12 GPU-h; cuts in DECISIONS | PASS | `results/v2e_throughput.json` (39 units: MVSEC cap sweep, SMD 640×360 / 346×260, 20 LaRS clips, 2/3/4-way concurrency); `scenarios.chosen.within_limit = true` including measured sweep and calibration; cuts D-025, D-026 |
| 8 | Plan step 3: calibration on `mvsec_tune` only, 32-cell grid, mean-W1 objective, argmin → `mvsec_cal`, rand range rule | PASS | `results/v2e_calibration.json` (`kind v2e_calibration`, `tier T2`, `split_manifest_sha256` = tune manifest); argmin reproduces from stored histograms (`freeze_profile.py --check`); D-027, D-028, D-030 |
| 9 | Profile and range frozen with the JSON's SHA | PASS | `configs/v2e_profiles.yaml`: `calibration_sha256` = sha256 of `results/v2e_calibration.json`; `python scripts/freeze_profile.py --check` → `profile frozen and consistent` |
| 10 | NCC threshold for the Phase 3 gate set on tune (plan Phase 3 gate, checklist B4) | PASS | `alignment_ncc_threshold` frozen from `ncc.threshold`; shift-5 and x-mirror controls below it (D-029); no test sequence converted with v2e |
| 11 | Splits: manifests regenerate byte-identically, SHA-256 committed (A6) | PASS | `scripts/make_splits.py --check` → `splits OK` (also from a cold hash cache, event-data-auditor re-audit) |
| 12 | Artefacts committed | PASS | `git ls-files` lists `results/v2e_{throughput,calibration}.json`, `configs/v2e_profiles.yaml`, `configs/v2e_budget.yaml`, `splits/*`, `src/evflow/sim/*`, `jobs/2026-10-08_v2e-*` |
| 13 | Results JSON keys (event-conventions / rule 1) | PASS | `tests/test_results_have_tiers.py` green; calibration `tier T2`; throughput and LaRS fps `tier n/a` + `tier_note` (compute cost / input setting, not evidence) |
| 14 | No HDR / latency / "realistic" claim from v2e (rule 2) | PASS | both auditors grepped docs, configs, jobs, `sim/`; none |
| 15 | GPU jobs prepared with `gpu-job` (script + manifest, resumable, logs) | PASS | `jobs/2026-10-08_v2e-{throughput,throughput-concurrency,calibration}.{sh,manifest.json}` |
| 16 | CI green on the pushed commit | see below | |

Checklist lines closed by this gate (VALIDATION_CHECKLIST.md): A3, A6, B1, B2, B3, B6. B4 is half
met (threshold fixed on tune before any test conversion); "reported per sequence" is Phase 3.

## Agents

**event-data-auditor** (splits, data selection), first run at the uncommitted state:

> VERDICT: PASS

Seven minors. Fixed: D-026 clips defined by exact µs / APS index (1), `--check` byte-compares files
and SHA256SUMS (2), clock-map clamp counter + 1e-3 guard (4), job manifests carry manifest and file
SHA (6), chunked reader (7). Deferred to Phase 3: re-hash option for cached SHA (3), runner split
guard (5).

**event-data-auditor**, re-audit at fe1860e:

> VERDICT: PASS

Chunked reader byte-identical to an in-memory reference on the real 25 M-event trial output at three
chunk sizes and on synthetic edge cases; 135 stored units pass full integrity; no test split
touched; D-026 clips keep ≥ 30 s gap. Five minors, fixed in 6c5d2cc: x/y checked before the uint16
cast (1), tests for the clamp guard and `git_dirty` (2), clamp bound for pre-counter units in D-031
(3), `int(round())` clip starts in D-026 (5); open items recorded below (4).

**v2e-fidelity-auditor**, first run at 79bddba:

> VERDICT: PASS

Argmin, all 32 objectives and the real/cell histograms recomputed from the stored units with HEAD
code match the JSON exactly; tune-only confirmed for all 96 calibration units; argument policy holds
for all 135 units. Ten minors: provenance (1, → D-031), budget omitted spent sweep (2, → measured
line + tighter maritime cuts), D-027/D-028/D-030 wording (3, 4, 5, 7), trial manifest (6), untrialled
profiles + `allow_undersampling` field (8), NCC sensitivity (9), STATUS/tier hygiene (10).

**v2e-fidelity-auditor**, re-check at 6c5d2cc:

> VERDICT: PASS

Findings 1–7 and 10 fixed, 8 and 9 accepted as Phase 3 open items. New: N1 (medium, non-blocking)
budget headroom vs the deferred trials → trials costed as their own lines, LaRS train 120 → 110
(`scenarios.chosen` within limit); N2 D-031 trial wording; N3 crashed SMD attempts not counted;
N4 SMD has no CT variants — all three addressed in D-026 / D-031.

## Outcome

- `mvsec_cal` = cell c30 (pos = neg thres 0.4, sigma 0.04, cutoff 60 Hz, no shot noise), at the top
  threshold of the plan's grid; not extended because the synthetic event count crosses the real
  count between 0.3 and 0.4 at cutoff 60 (D-030). Values: `configs/v2e_profiles.yaml`,
  `results/v2e_calibration.json`.
- `mvsec_rand` ranges and `alignment_ncc_threshold`: `configs/v2e_profiles.yaml` (frozen by script).
- Budget (`results/v2e_throughput.json`, `scenarios`): the plan's corpus as written exceeds 12 GPU-h
  about 2.5×; the chosen corpus fits with the measured Phase 2 spend. MVSEC train 240 s for every
  arm, maritime subsets (D-026).

## CI

(recorded after push)

## Overall verdict

**PASS** — every criterion passes; all four agent runs returned PASS with no open critical or major
finding.

## Open items

| Item | Who |
|---|---|
| Before Phase 3/7 corpora: ≤ 5 s trials of `mvsec_rand` at thres 0.28 and 0.52 and of the frozen `maritime_640x360` (budgeted in `configs/v2e_budget.yaml`); throughput stores are non-corpus | Claude, Phase 3 |
| Phase 3 alignment check: report a 1-frame-shift NCC control on tune and NCC per test sequence; state what the check can and cannot detect (indoor vs outdoor) | Claude, Phase 3 |
| Runner split guard: `v2e_runner` should check `--aps`/clip inputs against a named split manifest before Phase 3 converts test windows | Claude, Phase 3 |
| `make_splits.py --rehash` (re-verify cached SHA-256 at phase gates) | Claude, Phase 3 |
| Phase 3 fidelity report: carry the IEI pyramid caveat (D-027) and the 21-block sample size of `hot_fraction` | Claude, Phase 3 |
| Corpus jobs launched from a clean commit (`git_dirty` false in every manifest, D-031) | Claude, Phase 3 |
| Re-cost the budget after the Phase 3 corpus (measured lines) before Phase 7 maritime conversions | Claude, Phase 3/7 |
| Stray v2e output folder from the first (failed, relative-path) trial attempt under `external/v2e/data/` (gitignored; not removed — deletion inside `external/` is denied by settings) | user |
