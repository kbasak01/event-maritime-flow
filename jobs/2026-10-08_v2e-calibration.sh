#!/usr/bin/env bash
# Phase 2 step 3 (IMPLEMENTATION_PLAN.md): "Calibration (sim/calibrate.py) on mvsec_tune only: grid
# over pos_thres = neg_thres ∈ {0.15, 0.2, 0.3, 0.4}, sigma_thres ∈ {0.02, 0.04}, cutoff_hz ∈ {0, 60},
# shot noise ∈ {0, 1 Hz}; objective = mean Wasserstein-1 over the §1.6 event statistics vs real DVS
# on the same slice. Pick the argmin as profile mvsec_cal; ... Write results/v2e_calibration.json."
# Segments, cap and concurrency: configs/v2e_profiles.yaml calibration_grid + D-025/D-028.
# Every range is checked against splits/mvsec_tune.json inside calibrate.py (LeakageError).
# Resumable: finished cells (stats/<cell>.json) and units are skipped.
#   nohup bash jobs/2026-10-08_v2e-calibration.sh > runs/v2e-calibration/job.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WORK=data/converted/v2e/calibration
LOGS=runs/v2e-calibration
CONCURRENCY=${CONCURRENCY:-4}     # parallel cells on the one GPU (D-025: 4-way = 0.48 x sequential)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$WORK" "$LOGS"

echo "[$(date -Is)] real-event statistics"
$PY -m evflow.sim.calibrate real --work "$WORK" > "$LOGS/real.log" 2>&1
tail -1 "$LOGS/real.log"

echo "[$(date -Is)] grid cells, $CONCURRENCY at a time"
$PY -m evflow.sim.calibrate cells | xargs -P "$CONCURRENCY" -I{} bash -c \
  "$PY -m evflow.sim.calibrate cell --cell {} --work $WORK > $LOGS/{}.log 2>&1 \
   && echo \"[\$(date -Is)] {} \$(tail -1 $LOGS/{}.log)\" || echo \"[\$(date -Is)] {} FAILED\""

echo "[$(date -Is)] summarise"
$PY -m evflow.sim.calibrate summarise --work "$WORK"
echo "[$(date -Is)] done"
