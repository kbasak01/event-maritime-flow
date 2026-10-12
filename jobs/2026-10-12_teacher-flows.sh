#!/usr/bin/env bash
# Phase 7 step 2 (IMPLEMENTATION_PLAN.md): "teacher/raft.py: torchvision RAFT-large on the source
# frames resized to the event resolution, forward and backward, consistency mask (§1.6), stored
# per window." Units: smd_train, smd_test, lars_train, lars_eval_ct100 of the maritime corpus
# (D-054); teacher and mask definition D-055. Needs the v2e-maritime job's unit manifests (the
# frames are checked against their input SHA-256). Resumable: existing teacher stores are skipped.
#   nohup bash jobs/2026-10-12_teacher-flows.sh > runs/teacher-flows/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-12_teacher-flows.sh     # one smd_test unit, no summary
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
export PYTHONPATH=$PWD/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # D-052
LOGS=${LOGS:-$MAIN/runs/teacher-flows}
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"
if [ "${SMOKE:-0}" = 1 ]; then
  $PY scripts/run_teacher.py --corpus smd_test --limit 1 --no-summary
  exit 0
fi
for c in smd_test lars_eval_ct100 smd_train lars_train; do
  echo "[$(date -Is)] teacher: $c"
  $PY scripts/run_teacher.py --corpus "$c" --no-summary > "$LOGS/$c.log" 2>&1
  echo "[$(date -Is)] $c: $(grep -c '"key"' "$LOGS/$c.log") units"
done
echo "[$(date -Is)] summarise"
$PY scripts/run_teacher.py --summarise-only
echo "[$(date -Is)] done"
