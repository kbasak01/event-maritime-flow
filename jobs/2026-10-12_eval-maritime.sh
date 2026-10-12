#!/usr/bin/env bash
# Phase 7 steps 4-6 (IMPLEMENTATION_PLAN.md): "Evaluate every model (M0a, M0b, all MVSEC arms,
# maritime arms, R1 resized) on smd_test and lars_eval: T3a teacher-EPE (with mask coverage) and
# T3b FWL/RSAT, sliced by class on LaRS. Also re-evaluate the maritime arms on mvsec_test", the
# sea-clutter study, tables and figures. Matrix: src/evflow/eval/study_maritime.py (D-057);
# clutter: scripts/run_clutter.py (D-058). Frozen checkpoints only; nothing is chosen.
# Resumable: every unit skips if its JSON exists.
#   nohup bash jobs/2026-10-12_eval-maritime.sh > runs/eval-maritime/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-12_eval-maritime.sh    # 20 windows of two units -> runs/eval-maritime/smoke
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
export PYTHONPATH=$PWD/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # D-052
LOGS=${LOGS:-$MAIN/runs/eval-maritime}
RUNS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['runs'])")
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"
if [ "${SMOKE:-0}" = 1 ]; then
  $PY scripts/run_eval_maritime.py --model cm --dataset lars --sequence lars_eval_kf --max-windows 20
  $PY scripts/run_eval_maritime.py --model eraft --dataset smd --sequence MVI_0789_VIS_OB --max-windows 20
  exit 0
fi
i=0
$PY -m evflow.eval.study_maritime "$RUNS/train" > "$LOGS/commands.txt"
n=$(wc -l < "$LOGS/commands.txt")
while read -r cmd; do
  i=$((i + 1))
  name=$(echo "$cmd" | tr ' /' '__' | cut -c1-180)
  if $PY $cmd > "$LOGS/$name.log" 2>&1; then
    echo "[$(date -Is)] $i/$n $(tail -1 "$LOGS/$name.log" | cut -c1-200)"
  else
    echo "[$(date -Is)] $i/$n FAILED: $cmd (see $LOGS/$name.log)"
  fi
done < "$LOGS/commands.txt"
echo "[$(date -Is)] clutter study (CPU)"
$PY scripts/run_clutter.py
echo "[$(date -Is)] summarise"
$PY scripts/summarise_maritime.py || echo "[$(date -Is)] summary reports missing units"
$PY scripts/make_tables.py
$PY scripts/make_figures.py --bars-only
$PY scripts/make_figures.py --maritime-panels-only || echo "[$(date -Is)] panels failed"
echo "[$(date -Is)] done"
