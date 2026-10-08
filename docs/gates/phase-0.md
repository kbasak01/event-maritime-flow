# Phase 0 gate — Scaffold, environments, licences, data access

Date: 2026-10-08 · Gate run by: Claude (`/phase-gate 0`)

> **Exit gate:** `make lint test smoke` green locally and in CI; `python -c "import evflow"` in
> `evflow`; `v2e -h` (or `python v2e.py -h`) runs in `v2e`; `docs/LICENSES.md` has no **(verify)** left.

## Criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `make lint` green locally | PASS | `ruff check src tests scripts` → `All checks passed!` |
| 2 | `make test` green locally | PASS | `pytest -q` → `30 passed in 0.10s` |
| 3 | `make smoke` green locally | PASS | `ok` · `13 passed in 0.02s` |
| 4 | Lint + tests green in CI | PASS | GitHub Actions `ci` run 37845917094 on commit 4f103f7: `ruff check` ✓, `pytest -q` ✓ — https://github.com/kbasak01/event-maritime-flow/actions/runs/37845917094 |
| 5 | `python -c "import evflow"` in `evflow` (`.venv`, D-005) | PASS | `ok`; `torch 2.6.0+cu124 True NVIDIA RTX A4000`, ORT 1.30.0 providers `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']` |
| 6 | `python v2e.py -h` runs in `v2e` (`external/venvs/v2e`) | PASS | `usage: v2e.py [-h] [-o OUTPUT_FOLDER] ...`; v2e commit `cac638444557a224ba6ee5c29260916f152c00bd` (`configs/v2e_commit.txt`); SuperSloMo39.ckpt present (D-010) |
| 7 | `docs/LICENSES.md` has no (verify) | PASS | `grep -c "(verify)" docs/LICENSES.md` → `0` |
| 8 | `scripts/check_data.py` → `results/data_inventory.json` | PASS | present: `lars.root`, `v2e.slomo_ckpt`; absent: MVSEC, DSEC, SMD, E-RAFT checkpoints/preprocessed (not yet downloaded — open item) |
| 9 | Results JSON keys (event-conventions) | PASS | only `results/data_inventory.json` exists; `kind: data_inventory` (non-metric), checked by `tests/test_results_have_tiers.py` |
| 10 | G1 no GPL code (VALIDATION_CHECKLIST) | PASS | `tests/test_no_gpl_imports.py` 4 passed; reviewer grep clean |
| 11 | G2 no data in git | PASS | 31 files staged, largest 8 KB; `LARS Dataset/`, `external/`, `.venv/`, `*.zip`, `*:Zone.Identifier` ignored (`git check-ignore -v`) |
| – | `make test-data` | N/A | no `dataset`-marked tests yet (`30 deselected`, pytest exit 5); Phase 0 added no io/sim/eval code |

## Agents

**repo-publication-reviewer** (licence table only + light data check):

> VERDICT: SHIP

Findings (none blocking):
1. Minor — no LICENSES.md row for the E-RAFT checkpoints (`dsec.tar`, `mvsec_20.tar`, `mvsec_45.tar`);
   add a "model (external, not redistributed)" row with the weights' terms and training data (CC BY-SA 4.0) before Phase 4.
2. Minor — E-RAFT and tudelft/event_flow rows were dated from SOURCES.md, not re-read; re-check their
   LICENSE files when vendoring (Phase 1, Phase 4) and copy the header.
3. Info — `rpg_vid2e / ESIM` stays "check before use" while it is stretch-only.

Open questions raised by the reviewer (for the user):
- LaRS imagery in a public README: the terms allow images "in publications" but forbid "direct sharing";
  decide (or email the LaRS authors) before Phase 7 figures. D-008 currently allows it.
- Licence of released checkpoints trained on CC BY-NC (LaRS) / CC BY-SA (MVSEC, DSEC) derived data — decide before Phase 10.
- SMD: ask D. K. Prasad for written permission for imagery, or keep SMD out of the README permanently.
- SuperSloMo39.ckpt has no stated licence: never bundle it in a release.

## CI

PASS — run 37845917094 (https://github.com/kbasak01/event-maritime-flow/actions/runs/37845917094), conclusion `success` on 4f103f7. Annotations (non-blocking):
Node.js 20 deprecation for `actions/checkout@v4` / `actions/setup-python@v5`; `ubuntu-latest` moves to
Ubuntu 26 from 2026-10-19 — revisit action versions if CI breaks.

## Open items

| Item | Who |
|---|---|
| Download MVSEC (`outdoor_day1/2`, `indoor_flying1–3`, GT flow, calib), DSEC-Flow (2–3 train seqs), SMD VIS on-board, E-RAFT checkpoints + preprocessed MVSEC; set their paths in `configs/paths.local.yaml`; re-run `python scripts/check_data.py` | user (download), Claude (inventory) |
| LaRS sequences zip (13.6 GB) still to download | user |
| Install ffmpeg/ffprobe (`sudo apt install ffmpeg`) before Phase 2 (`--input_frame_rate` from ffprobe) | user |
| Reviewer findings 1–2 (E-RAFT checkpoint row; re-read MIT licences when vendoring) | Claude, Phases 1 & 4 |
| Reviewer open questions (LaRS README imagery, checkpoint licence, SMD permission) | user decision before Phases 7/10 |

## Overall verdict

**PASS** — every criterion passes (1–11), repo-publication-reviewer SHIP, CI green.
