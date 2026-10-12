#!/usr/bin/env bash
# Phase 7 step 1 (IMPLEMENTATION_PLAN.md): "v2e(SMD on-board) and v2e(LaRS sequences: 9 context
# frames + keyframe) with mvsec_cal and two CT variants (x0.67, x1.5)."
# Units: src/evflow/sim/maritime_corpus.py from configs/v2e_corpus_maritime.yaml (D-026, D-054):
# 2 trials, smd_train 8, smd_test 3, lars_train 110, lars_eval_ct{067,100,150} 3 x 40.
# Resumable: finished units (store + manifest) are skipped by the runner.
# Launch from a clean commit (D-031), a detached worktree with paths.local.yaml copied in:
#   nohup bash jobs/2026-10-12_v2e-maritime.sh > runs/v2e-maritime/job.log 2>&1 &
#   TRIALS_ONLY=1 bash jobs/2026-10-12_v2e-maritime.sh      # the two <= 5 s trials, then stop
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
export PYTHONPATH=$PWD/src          # this checkout's code, not the editable install's
LOGS=${LOGS:-$MAIN/runs/v2e-maritime}
CONCURRENCY=${CONCURRENCY:-4}       # D-025: 4-way = 0.48 x sequential
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"

echo "[$(date -Is)] plan"
$PY -m evflow.sim.maritime_corpus plan

run_unit() {   # $1 = <corpus>/<name>
  local log="$LOGS/${1//\//_}.log"
  if $PY -m evflow.sim.maritime_corpus unit --key "$1" > "$log" 2>&1; then
    echo "[$(date -Is)] $1 $(tail -1 "$log")"
  else
    echo "[$(date -Is)] $1 FAILED (see $log)"
  fi
}
export -f run_unit
export PY LOGS

echo "[$(date -Is)] trials (sequential)"
for k in $($PY -m evflow.sim.maritime_corpus units --corpus trial); do run_unit "$k"; done
if [ "${TRIALS_ONLY:-0}" = 1 ]; then echo "[$(date -Is)] trials only: stop"; exit 0; fi

echo "[$(date -Is)] corpus units, $CONCURRENCY at a time"
$PY -m evflow.sim.maritime_corpus units --exclude trial | xargs -P "$CONCURRENCY" -I{} bash -c 'run_unit {}'

echo "[$(date -Is)] summarise"
$PY -m evflow.sim.maritime_corpus summarise
echo "[$(date -Is)] done"
