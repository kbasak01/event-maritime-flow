#!/usr/bin/env bash
# Phase 3 gate fix, D-035 (IMPLEMENTATION_PLAN Phase 2 step 3 rule, applied with a timing
# constraint): the Phase 3 v2e-fidelity audit found v2e events ~0.5 APS frame late with the
# frozen cutoff_hz 60 (tune evidence). Re-calibrate on mvsec_tune ONLY: grid
# calibration_grid_timing (cutoff_hz 0, thres 0.4-0.7, sigma 0.02/0.04, shot 0/1 Hz; same 3 tune
# segments and seed as Phase 2), lag profile for every cell of both grids, argmin of the Phase 2
# objective among lag-eligible cells -> results/v2e_calibration_timing.json.
# Every range is checked against splits/mvsec_tune.json inside calibrate.py (LeakageError).
# Resumable: finished cells (stats/<cell>.json) and units are skipped.
# Launch from a clean commit (D-031), e.g. a detached worktree with paths.local.yaml copied in:
#   nohup bash jobs/2026-10-09_v2e-calibration-timing.sh > runs/v2e-calibration-timing/job.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
export PYTHONPATH=$PWD/src          # this checkout's code, not the editable install's
EVENTS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['events'])")
WORK=$EVENTS/v2e/calibration_timing
BASE=$EVENTS/v2e/calibration        # Phase 2 grid (cells c00-c31), reused, not re-run
LOGS=${LOGS:-$MAIN/runs/v2e-calibration-timing}
CONCURRENCY=${CONCURRENCY:-4}
GRID=calibration_grid_timing
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$WORK" "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"

echo "[$(date -Is)] real-event statistics"
$PY -m evflow.sim.calibrate real --work "$WORK" --grid $GRID > "$LOGS/real.log" 2>&1
tail -1 "$LOGS/real.log"

echo "[$(date -Is)] grid cells, $CONCURRENCY at a time"
$PY -m evflow.sim.calibrate cells --grid $GRID | xargs -P "$CONCURRENCY" -I{} bash -c \
  "$PY -m evflow.sim.calibrate cell --grid $GRID --cell {} --work $WORK > $LOGS/{}.log 2>&1 \
   && echo \"[\$(date -Is)] {} \$(tail -1 $LOGS/{}.log)\" || echo \"[\$(date -Is)] {} FAILED\""

echo "[$(date -Is)] summarise (lag profiles for both grids, choice among lag-eligible cells)"
$PY -m evflow.sim.calibrate summarise-timing --work "$WORK" --base-work "$BASE"
echo "[$(date -Is)] done"
