# STATUS

| Phase | State | Gate file | Date |
|---|---|---|---|
| 0 | PASS (CI green) | docs/gates/phase-0.md | 2026-10-08 |
| 1 | PASS (CI green) | docs/gates/phase-1.md | 2026-10-08 |
| 2 | PASS (CI green) | docs/gates/phase-2.md | 2026-10-09 |
| 3 | PASS (CI green) | docs/gates/phase-3.md | 2026-10-09 |
| 4 | PASS (CI green) | docs/gates/phase-4.md | 2026-10-10 |
| 5 | PASS (CI green) | docs/gates/phase-5.md | 2026-10-11 |
| 6 | PASS (CI green) | docs/gates/phase-6.md | 2026-10-12 |
| 7 | PASS (CI green) | docs/gates/phase-7.md | 2026-10-12 |
| 8 | not started | – | – |
| 9 | not started | – | – |
| 10 | not started | – | – |

Pending GPU jobs: none (Claude may launch GPU jobs after `gpu-job` prep, D-006; one CUDA process at a time on this workstation, D-048b).

Done GPU jobs (Phase 7):
- `jobs/2026-10-12_v2e-maritime.sh` → `results/v2e_maritime_corpus.json` (243/243 units, clean at 7fd867c, 11:04–14:02 on 2026-10-12 incl. trials; measured 2.89 GPU-h, budget 19.44/19.5 after the user accepted the overrun, D-059).
- `jobs/2026-10-12_teacher-flows.sh` → `results/teacher/summary.json` (161 units, clean at cdc358b, 14:05–14:22).
- `jobs/2026-10-12_train-maritime.sh` → `results/train/maritime_summary.json` (2/2 runs at b552569, 14:26–16:11; attempt 1 at 4164c3d failed at epoch 0, D-060 amendment, `runs/train-maritime/attempt1/`).
- `jobs/2026-10-12_eval-maritime.sh` → `results/eval_maritime/` (176/176 units), `results/eval/maritime_*/`, `results/maritime/clutter.json`, `results/TABLES.md`, `results/figures/clutter_ct.png` (main checkout from 5eea7f9, 16:13–17:56; summary re-run after the block-length fix at 0e85377).

Done GPU jobs (Phase 6):
- `jobs/2026-10-11_eval-all.sh` → `results/eval/` (298/298 units incl. Phase 4 baselines and Phase 5 gate units; reproduction unit equal to its committed JSON, max diff 0), `results/eval/{summary,aggregate,paired}.json`, `results/TABLES.md`, `results/figures/`. Attempt 1 stopped after 8 min (CPU threads, D-052); attempt 2 at b19e609, 16:31–22:47 on 2026-10-11 (`runs/eval-all/job.log`).

Done GPU jobs (Phase 5):
- `jobs/2026-10-10_phase5-train.sh` → `runs/train/` (17 runs, all TRUSTWORTHY), `runs/train-sweep/` (8 CM cells), `results/train/{summary,cm_sweep}.json`, `results/eval/real_sup/*/` gate units (attempts 1–3, D-045/D-047/D-048; final launch at 7904922 from a clean worktree, ended 2026-10-11 13:28).

Done GPU jobs (Phase 4):
- `jobs/2026-10-09_phase4-baselines.sh` → `results/baselines/` (reproduction gate, M0b selection on `mvsec_tune`, 33 baseline units; at 0da94b3, wall time in `runs/phase4-baselines/job.log`).
- `scripts/diagnose_r1_mvsec.py` → `results/baselines/r1_mvsec_diagnostic.json` (b9346c5, D-042).

Done GPU jobs (Phase 3):
- `jobs/2026-10-09_v2e-mvsec-d035.sh` → `results/v2e_mvsec_corpus.json` (D-035 corpus, 30 units at b6853e6; interrupted once by a user pause, D-036).
- `jobs/2026-10-09_v2e-calibration-timing.sh` → `results/v2e_calibration_timing.json` (D-035, 16 tune cells).
- `jobs/2026-10-09_v2e-mvsec.sh` → `results/v2e_mvsec_corpus_d032_lagged.json` (30 units, 5.67 h wall, 97210ab; lagged profile, superseded by D-035).

Done GPU jobs (Phase 2):
- `jobs/2026-10-08_v2e-throughput.sh` + `jobs/2026-10-08_v2e-throughput-concurrency.sh` → `results/v2e_throughput.json`.
- `jobs/2026-10-08_v2e-calibration.sh` → `results/v2e_calibration.json` (measured GPU time in `results/v2e_throughput.json`, `scenarios.chosen`).

Pending data: none. All datasets are present in `data/` (results/data_inventory.json). ffmpeg is installed.
