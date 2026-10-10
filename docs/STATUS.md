# STATUS

| Phase | State | Gate file | Date |
|---|---|---|---|
| 0 | PASS (CI green) | docs/gates/phase-0.md | 2026-10-08 |
| 1 | PASS (CI green) | docs/gates/phase-1.md | 2026-10-08 |
| 2 | PASS (CI green) | docs/gates/phase-2.md | 2026-10-09 |
| 3 | PASS (CI green) | docs/gates/phase-3.md | 2026-10-09 |
| 4 | PASS (CI green) | docs/gates/phase-4.md | 2026-10-10 |
| 5 | not started | – | – |
| 6 | not started | – | – |
| 7 | not started | – | – |
| 8 | not started | – | – |
| 9 | not started | – | – |
| 10 | not started | – | – |

Pending GPU jobs: none (Claude may launch GPU jobs after `gpu-job` prep, D-006).

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
