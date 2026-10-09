# STATUS

| Phase | State | Gate file | Date |
|---|---|---|---|
| 0 | PASS (CI green) | docs/gates/phase-0.md | 2026-10-08 |
| 1 | PASS (CI green) | docs/gates/phase-1.md | 2026-10-08 |
| 2 | PASS (CI green) | docs/gates/phase-2.md | 2026-10-09 |
| 3 | in progress | – | – |
| 4 | not started | – | – |
| 5 | not started | – | – |
| 6 | not started | – | – |
| 7 | not started | – | – |
| 8 | not started | – | – |
| 9 | not started | – | – |
| 10 | not started | – | – |

Pending GPU jobs (Claude may launch GPU jobs after `gpu-job` prep, D-006):
- `jobs/2026-10-09_v2e-mvsec.sh` (Phase 3 step 1, 30 units, est. 13.3 sequential GPU-h ≈ 6.5 h at 4-way, ≈ 15 GB) → `results/v2e_mvsec_corpus.json`.

Done GPU jobs (Phase 2):
- `jobs/2026-10-08_v2e-throughput.sh` + `jobs/2026-10-08_v2e-throughput-concurrency.sh` → `results/v2e_throughput.json`.
- `jobs/2026-10-08_v2e-calibration.sh` → `results/v2e_calibration.json` (measured GPU time in `results/v2e_throughput.json`, `scenarios.chosen`).

Pending data: none. All datasets are present in `data/` (results/data_inventory.json). ffmpeg is installed.
