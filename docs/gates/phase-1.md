# Phase 1 gate — Event core: store, slicing, representations, metrics, IWE/FWL

Date: 2026-10-08 · Gate run by: Claude (`/phase-gate 1`) · Code commit: 1d1b782

> **Exit gate:** all tests green; `flow-metric-verifier` **PASS**; `event-data-auditor` **PASS** on
> the converters run against one real MVSEC and one real DSEC file (marked `dataset`).

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `make lint` green | PASS | `ruff check src tests scripts` → `All checks passed!` |
| 2 | `make test` green (CPU, no data) | PASS | `162 passed, 4 deselected in 2.46s` |
| 3 | `make test-data` green on one MVSEC + one DSEC file | PASS | `4 passed, 162 deselected in 5.89s` (`tests/test_io_dataset.py`: `mvsec_indoor_flying1_left.h5`, `dsec_thun_00_a_left.h5`) |
| 4 | `make smoke` still green | PASS | `13 passed in 0.02s` |
| 5 | Analytic fixtures pass (plan step 6) | PASS | `tests/fixtures/moving_edge.py`; `tests/test_iwe.py` 21 passed. Zero-flow FWL = RSAT = 1.0 exactly; correct flow FWL 7.07 vs wrong sign 1.01 (round) |
| 6 | Slicer: DSEC docstring example, empty windows, past-end `None` | PASS | `tests/test_store.py` (19 tests). `ms_to_idx` length rule `t[-1]//1000+2` matches the real DSEC file exactly (D-016) |
| 7 | Representations: mass conservation, bin edges, torch == numpy ≤ 1e-6 | PASS | `tests/test_representations.py` (26) |
| 8 | Metrics: hand 4×4 cases, masks, %Out needs both conditions | PASS | `tests/test_metrics.py` (13, incl. float32 input) |
| 9 | DSEC flow PNG round trip | PASS | `tests/test_png.py` (4, incl. an independent RGB PNG writer for channel order) |
| 10 | Oracle parity (MIT tudelft/event_flow) to 1e-6 | PASS | `tests/test_oracle_parity.py` (20): FWL, RSAT, IWE round/bilinear at t_ref 0/0.5/1, AEE, %Out on \|pred\| = \|GT\| fields. Deliberate %Out definition difference pinned (D-020) |
| 11 | Layouts inspected and recorded before each converter | PASS | D-015 (MVSEC), D-016 (DSEC); `scripts/h5_layout.py` stands in for `h5ls -v` (not installed) |
| 12 | Artefacts committed | PASS | `git ls-files` lists `src/evflow/{io,events,flow}/*`, `tests/{fixtures,oracles}/*`, `configs/eval_mvsec.yaml` (commit 1d1b782) |
| 13 | Results JSON keys (event-conventions) | PASS | no new results JSON in Phase 1; only `results/data_inventory.json` (non-metric) |
| 14 | G1 no GPL code / G2 no data in git | PASS | `tests/test_no_gpl_imports.py` green; staged files contain no `.h5/.hdf5/.npz/.png/.zip` or `data/` paths; largest file 20 KB (oracle) |
| 15 | CI green on the pushed commit | PASS | GitHub Actions `ci` run 37858459879 on d25b228: `ruff check` → `All checks passed!`, `pytest -q` → `162 passed, 4 deselected in 2.18s` |

Checklist lines closed by this gate (VALIDATION_CHECKLIST.md): A1, A2, A4, A5, C1, C2. C3 and C4 are
met for the code and config; the README quote of C4 is Phase 10.

## Agents

**flow-metric-verifier**, first run:

> VERDICT: FAIL

Its one major finding was in the protocol config, not the code. `configs/eval_mvsec.yaml` and D-019(d) said the
outdoor_day1 window "= E-RAFT range(4356, 4706) of timestamps_flow.txt". E-RAFT, however, uses
`align_to: depth`, and `timestamps_depth[i] == timestamps_flow[i+1]`, so E-RAFT's event windows
are one GT interval (50 ms) later. The Phase 4 E-RAFT pairing was not declared. There were three minors:
- the sparse mask deviates from EV-FlowNet's uint8-rescaled count image, and this was not recorded;
- the torch `ae_deg` lost about 1.7e-3° in float32;
- the agreement figures with EV-FlowNet were not noted.

Fixes:
- the config comment and D-019(d) were reworded, and the `reproduction_eraft` block now declares E-RAFT's windows, GT, valid rule, crop and dense mask;
- D-019(b) records the sparse-mask deviation;
- `flow_metrics` (torch) computes in float64, with a new float32 test;
- D-019(e) cites the agreement figures.

**flow-metric-verifier**, re-run:

> VERDICT: PASS

All earlier findings resolved. The independent hand cases, fixture FWL/RSAT values, oracle verbatim
check and GT-propagation comparison against EV-FlowNet's own function were all re-run. The dt=1 maximum
difference is 1.2e-4 px, and dt=4 differs by at most 2.8e-4 px on any per-window mean. Carried forward:
flip augmentation (u ← −u) is checked when the augmentation code lands (Phase 5).

**event-data-auditor**:

> VERDICT: PASS

- **A1:** both stores pass full integrity checks.
  - indoor_flying1: 14,071,304 events, 70.29 s, 0.200 Mev/s.
  - thun_00_a: 134,133,270 events, 11.90 s, 11.27 Mev/s.
  - `ms_to_idx` matches the DSEC definition at random ms and both ends, and equals a full recompute.
  - For DSEC it is also identical to the source.
- **Element-wise comparison against the sources** finds 0 mismatches over all events. The 71.5 % ON
  share of indoor_flying1 is a property of the source, not a mapping error: it holds across pixels and
  time, and the APS log-intensity correlation is positive in 38/38 frame pairs.
- **A4:** a physical check confirms the DSEC GT lives in the rectified frame (21/21 windows).
- **A5:** MVSEC GT and events share the raw grid.
- **Item 4:** no Phase 1 code or config selects data from a test split.
  - indoor_flying1 and thun_00_a were used only for converter verification.
  - Items 3 and 5 are N/A in Phase 1.

Minor findings and their status:
1. Rectification was opt-in, which could silently feed raw coordinates later. **Fixed:**
   `EventSlicer.get_events` now rectifies by default when the store has a `rectify_map`, and raw
   coordinates need `rectify=False`. Tests, CONVENTIONS and D-017 are updated. This change came
   after the audit run; it is covered by `make test`/`make test-data`.
2. The store manifests recorded a dirty git state. **Fixed:** both stores were re-converted from
   clean commit 1d1b782, and their manifests now show `git_dirty: false`.
3. The DSEC store lacked mapping attrs. **Fixed:** `time_conversion` and `polarity_mapping` attrs
   were added and are tested.
4. The layout tool is a stand-in for `h5ls`. **Accepted:** use the same tool for A3 in Phase 2.
5. The polarity statistic of a test sequence must not feed Phase 2 calibration. **Carried forward** to
   Phase 2: the calibration manifest lists `mvsec_tune` sources only.

## CI

PASS — run 37858459879 (https://github.com/kbasak01/event-maritime-flow/actions/runs/37858459879), conclusion `success` on d25b228.

## Overall verdict

**PASS.** Every criterion passes, and both agents' latest verdicts are PASS with no open critical or major
finding.

## Open items

| Item | Who |
|---|---|
| Phase 4: define "E-RAFT's own evaluation" (the E-RAFT repo ships no metric code) and check whether the paper's MVSEC numbers are dense or sparse, before the C5 reproduction gate | Claude, Phase 4 |
| Phase 5: verify flip augmentation negates the right flow component | Claude, Phase 5 |
| Phase 2: v2e `--dvs_h5` layout via `scripts/h5_layout.py`, recorded before `io/v2e_reader.py` (A3); calibration manifest uses `mvsec_tune` only | Claude, Phase 2 |
| Convert the remaining MVSEC / DSEC sequences when Phases 2–4 need them (`scripts/convert_events.py`, seconds per file) | Claude |
