#!/usr/bin/env bash
# Phase 5: train the MVSEC arms + representation ablation. IMPLEMENTATION_PLAN.md Phase 5:
#   2. "/gpu-job train-<arm>-<seed> for every arm in §1.4 (MVSEC arms only). Selection on
#      mvsec_tune EPE (supervised arms) or FWL (self-supervised arms) only; checkpoint SHA-256
#      recorded."   3. "Representation ablation (§1.5)."
# Design: D-043 (mixed dt, own-stream selection, matched windows, hood rows); CM sweep D-044.
# Order (one script, CONCURRENCY units at a time where independent):
#   0. per-window GT caches for mvsec_train / mvsec_tune (CPU)
#   1. real_sup s1337 (+ v2e_fixed_sup s1337 alongside) -> gate unit: real_sup s1337 on
#      indoor_flying1 dt1; the job STOPS if its EPE > GATE_EPE (Phase 5 exit gate, D-043)
#   2. CM weight sweep (2 self-supervised arms x 4 cells, 10 epochs, own-stream tune FWL)
#      -> scripts/freeze_cm_sweep.py -> results/train/cm_sweep.json
#   3. pool: v2e_rand_sup s1337, real_selfsup, v2e_selfsup (frozen weights), seeds 2026/7 of the
#      3 supervised arms, ablation (voxel15, timesurface, count) x (real_sup, v2e_fixed_sup) s1337
#   4. gate units for real_sup seeds 2026/7; scripts/summarise_training.py
# Every training unit skips if its run has DONE and resumes from last.pt otherwise; eval units
# skip if their JSON exists. No unit trains or selects on a test split (split guard in
# src/evflow/train/data.py); test units only read frozen checkpoints.
# Launch from a clean commit in the main tree (code paths must stay unchanged while it runs):
#   nohup bash jobs/2026-10-10_phase5-train.sh > runs/phase5-train/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-10_phase5-train.sh   # 1 epoch, 40 train / 40 tune windows, runs/ only
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${EVFLOW_PY:-$PWD/.venv/bin/python}
export PYTHONPATH=$PWD/src
LOGS=${LOGS:-$PWD/runs/phase5-train}
CONCURRENCY=${CONCURRENCY:-2}       # one run uses ~50 % of the A4000 and ~5 GB
GATE_EPE=${GATE_EPE:-2.0}           # D-043: 2 x the upper literature value (~1.0 px)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"
RUNS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['runs'])")
REP=voxel5
SEEDS_EXTRA="2026 7"

if [ "${SMOKE:-0}" = 1 ]; then
  echo "[$(date -Is)] smoke: one epoch of real_sup and real_selfsup on 40 windows -> $RUNS/smoke-train"
  for arm in real_sup real_selfsup; do
    $PY -m evflow.train.loop --config configs/train_$arm.yaml --epochs 1 --max-train-windows 40 \
      --max-tune-windows 40 --run-root "$RUNS/smoke-train" > "$LOGS/smoke_$arm.log" 2>&1
    tail -n 1 "$LOGS/smoke_$arm.log"
  done
  echo "[$(date -Is)] smoke done"; exit 0
fi

train_unit() {   # name, then loop args
  local name="$1"; shift
  if $PY -m evflow.train.loop "$@" > "$LOGS/train_$name.log" 2>&1; then
    echo "[$(date -Is)] $name: $(tail -n 1 "$LOGS/train_$name.log")"
  else
    echo "[$(date -Is)] $name FAILED (see $LOGS/train_$name.log)"
  fi
}
eval_unit() {    # run_id sequence dt
  local name="eval_$1_$2_dt$3"
  if $PY scripts/run_eval.py --run-dir "$RUNS/train/$1" --dataset mvsec --sequence "$2" \
       --dt "$3" > "$LOGS/$name.log" 2>&1; then
    echo "[$(date -Is)] $(tail -n 1 "$LOGS/$name.log")"
  else
    echo "[$(date -Is)] $name FAILED (see $LOGS/$name.log)"
  fi
}
unit() {         # one line of a unit list: train <name> <args...> | sweep <arm> <sw> <dp>
  case "$1" in
    train) shift; local n="$1"; shift; train_unit "$n" "$@" ;;
    sweep) train_unit "sweep_$2_sw$3_dp$4" --config "configs/train_$2.yaml" \
             --epochs "$SWEEP_EPOCHS" --cm-smoothness "$3" --cm-divergence "$4" \
             --suffix "_sw$3_dp$4" --run-root "$RUNS/train-sweep" ;;
  esac
}
export -f train_unit eval_unit unit
export PY LOGS RUNS

echo "[$(date -Is)] stage 0: GT caches"
for split in mvsec_tune mvsec_train; do
  (cd scripts && $PY cache_mvsec_gt.py --split $split --dt 1 4) > "$LOGS/gt_$split.log" 2>&1
  tail -n 2 "$LOGS/gt_$split.log"
done

echo "[$(date -Is)] stage 1: real_sup s1337 (+ v2e_fixed_sup s1337), then the gate unit"
printf '%s\n' "train real_sup_${REP}_s1337 --config configs/train_real_sup.yaml --seed 1337" \
              "train v2e_fixed_sup_${REP}_s1337 --config configs/train_v2e_fixed_sup.yaml --seed 1337" \
  | xargs -P "$CONCURRENCY" -I{} bash -c 'unit {}'
eval_unit "real_sup_${REP}_s1337" indoor_flying1 1
GATE_JSON=results/eval/real_sup/1337/mvsec_indoor_flying1_dt1.json
EPE=$($PY -c "import json; print(json.load(open('$GATE_JSON'))['metrics']['epe'])")
echo "[$(date -Is)] gate unit: real_sup s1337 indoor_flying1 dt1 EPE $EPE (limit $GATE_EPE)"
if ! $PY -c "import sys; sys.exit(0 if $EPE <= $GATE_EPE else 1)"; then
  echo "[$(date -Is)] STOP: real_sup misses the Phase 5 gate; fix before training the other arms"
  exit 3
fi

echo "[$(date -Is)] stage 2: CM weight sweep"
SWEEP_EPOCHS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/evflow.yaml'))['losses']['cm']['sweep']['epochs'])")
export SWEEP_EPOCHS
$PY - <<'EOF' | xargs -P "$CONCURRENCY" -I{} bash -c 'unit {}'
import yaml
s = yaml.safe_load(open("configs/evflow.yaml"))["losses"]["cm"]["sweep"]
for arm in ("real_selfsup", "v2e_selfsup"):
    for sw in s["smoothness_weight"]:
        for dp in s["divergence_penalty"]:
            print(f"sweep {arm} {sw:g} {dp:g}")
EOF
$PY scripts/freeze_cm_sweep.py > "$LOGS/cm_sweep.log" 2>&1 || echo "[$(date -Is)] CM sweep: no eligible cell for an arm (see $LOGS/cm_sweep.log)"
cat "$LOGS/cm_sweep.log"

echo "[$(date -Is)] stage 3: remaining runs, $CONCURRENCY at a time"
{
  echo "train v2e_rand_sup_${REP}_s1337 --config configs/train_v2e_rand_sup.yaml --seed 1337"
  for arm in real_selfsup v2e_selfsup; do
    CW=$($PY -c "import json; c=json.load(open('results/train/cm_sweep.json'))['arms']['$arm']['chosen']; print('' if c is None else f\"--cm-smoothness {c['smoothness_weight']} --cm-divergence {c['divergence_penalty']}\")")
    if [ -n "$CW" ]; then
      echo "train ${arm}_${REP}_s1337 --config configs/train_$arm.yaml --seed 1337 $CW"
    fi
  done
  for seed in $SEEDS_EXTRA; do
    for arm in real_sup v2e_fixed_sup v2e_rand_sup; do
      echo "train ${arm}_${REP}_s$seed --config configs/train_$arm.yaml --seed $seed"
    done
  done
  for rep in voxel15 timesurface count; do
    for arm in real_sup v2e_fixed_sup; do
      echo "train ${arm}_${rep}_s1337 --config configs/train_$arm.yaml --seed 1337 --representation $rep"
    done
  done
} | xargs -P "$CONCURRENCY" -I{} bash -c 'unit {}'

echo "[$(date -Is)] stage 4: gate units for the other real_sup seeds, summary"
for seed in $SEEDS_EXTRA; do eval_unit "real_sup_${REP}_s$seed" indoor_flying1 1; done
EXPECT=""
for seed in 1337 $SEEDS_EXTRA; do
  for arm in real_sup v2e_fixed_sup v2e_rand_sup; do EXPECT="$EXPECT ${arm}_${REP}_s$seed"; done
done
EXPECT="$EXPECT real_selfsup_${REP}_s1337 v2e_selfsup_${REP}_s1337"
for rep in voxel15 timesurface count; do
  for arm in real_sup v2e_fixed_sup; do EXPECT="$EXPECT ${arm}_${rep}_s1337"; done
done
$PY scripts/summarise_training.py --expect $EXPECT || true
echo "[$(date -Is)] done"
