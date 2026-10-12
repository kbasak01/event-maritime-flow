#!/usr/bin/env bash
# Phase 7 step 3 (IMPLEMENTATION_PLAN.md): "Train maritime_selfsup and maritime_teacher (one seed
# each)." Data: the calibrated maritime corpus smd_train + lars_train (D-054); maritime_teacher
# supervised by the RAFT teacher on FB-consistent pixels (D-055); maritime_selfsup with the
# average-timestamp CM loss, v2e_selfsup's chosen sweep weights (results/train/cm_sweep.json,
# mvsec_tune) and the low-count floor (D-049, D-056). Selection: v2e(mvsec_tune), dt 1 (D-056).
# Never reads smd_test / lars_eval / mvsec_test (split guard in train/data.py). Resumable
# (last.pt). One CUDA process at a time (D-048b).
#   nohup bash jobs/2026-10-12_train-maritime.sh > runs/train-maritime/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-12_train-maritime.sh   # 1 epoch on 64 windows each -> runs/smoke-train
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
export PYTHONPATH=$PWD/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGS=${LOGS:-$MAIN/runs/train-maritime}
RUNS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['runs'])")
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"
CW=$($PY -c "import json; c=json.load(open('results/train/cm_sweep.json'))['arms']['v2e_selfsup']['chosen']; print(f\"--cm-smoothness {c['smoothness_weight']} --cm-divergence {c['divergence_penalty']}\")")
echo "[$(date -Is)] maritime_selfsup CM weights (v2e_selfsup's sweep choice): $CW"
SMOKE_ARGS=""
if [ "${SMOKE:-0}" = 1 ]; then SMOKE_ARGS="--epochs 1 --max-train-windows 64 --max-tune-windows 64"; fi
for arm in maritime_teacher maritime_selfsup; do
  extra=""
  [ "$arm" = maritime_selfsup ] && extra="$CW"
  echo "[$(date -Is)] train $arm"
  if $PY -m evflow.train.loop --config "configs/train_$arm.yaml" $extra $SMOKE_ARGS \
       > "$LOGS/train_${arm}_voxel5_s1337${SMOKE:+_smoke}.log" 2>&1; then
    echo "[$(date -Is)] $arm: $(tail -1 "$LOGS/train_${arm}_voxel5_s1337${SMOKE:+_smoke}.log")"
  else
    echo "[$(date -Is)] $arm FAILED (see log)"
  fi
done
if [ "${SMOKE:-0}" = 1 ]; then exit 0; fi
echo "[$(date -Is)] summarise"
$PY scripts/summarise_training.py --runs "$RUNS/train" --phase 7 \
    --expect maritime_selfsup_voxel5_s1337 maritime_teacher_voxel5_s1337 \
    --out results/train/maritime_summary.json
echo "[$(date -Is)] done"
