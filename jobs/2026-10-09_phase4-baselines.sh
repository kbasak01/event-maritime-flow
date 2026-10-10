#!/usr/bin/env bash
# Phase 4: baselines and the reference reproduction gate. IMPLEMENTATION_PLAN.md Phase 4:
#   1. "M0a, M0b on mvsec_test (dt = 1, 4) and dsec_test."
#   2. "R1 (E-RAFT DSEC checkpoint) on dsec_test with its own 15-bin voxel / 100 ms setting."
#      (+ R1 = E-RAFT 20 Hz MVSEC checkpoint on mvsec_test, user request, D-040)
#   3. "Reproduction gate: R1's MVSEC 20 Hz checkpoint on outdoor_day1 using E-RAFT's
#      pre-processed MVSEC release ..." (definition D-039)
# Stages: (1) reproduction gate (E-RAFT's own pipeline in external/venvs/eraft, then our parity
# check); (2) M0b selection on mvsec_tune ONLY (9 grid cells in parallel, then the argmin JSON);
# (3) the 33 evaluation units (M0a, M0b, R1 x 8 MVSEC + 3 DSEC), CONCURRENCY at a time, longest
# first; (4) summariser. Every unit skips if its results JSON exists (resumable). No unit trains
# or selects anything on a test split: test units only read the frozen tune argmin.
# Launch from a clean commit (git_dirty is checked over src/scripts/configs/jobs/splits):
#   nohup bash jobs/2026-10-09_phase4-baselines.sh > runs/phase4-baselines/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-09_phase4-baselines.sh    # 5 windows per stage, writes runs/ only
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${EVFLOW_PY:-$PWD/.venv/bin/python}
ERAFT_PY=${ERAFT_PY:-$PWD/external/venvs/eraft/bin/python}
export PYTHONPATH=$PWD/src
LOGS=${LOGS:-$PWD/runs/phase4-baselines}
CONCURRENCY=${CONCURRENCY:-6}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"
paths() { $PY -c "import yaml; d=yaml.safe_load(open('configs/paths.local.yaml')); print($1)"; }

if [ "${SMOKE:-0}" = 1 ]; then
  echo "[$(date -Is)] smoke: 5 windows per model/dataset, tune 5 windows, outputs under runs/"
  $PY scripts/tune_cm_baseline.py --max-windows 5 --only-cell 0 > "$LOGS/smoke_tune.log" 2>&1
  for a in "zero --dataset mvsec --sequence indoor_flying1 --dt 1" \
           "eraft --dataset mvsec --sequence outdoor_day1 --dt 4" \
           "eraft --dataset dsec --sequence thun_00_a" "zero --dataset dsec --sequence thun_00_a"; do
    $PY scripts/run_baselines.py --model $a --max-windows 5 --force >> "$LOGS/smoke_units.log" 2>&1
  done
  tail -n 2 "$LOGS/smoke_tune.log" "$LOGS/smoke_units.log"
  echo "[$(date -Is)] smoke done"; exit 0
fi

echo "[$(date -Is)] stage 1: reproduction gate"
REPRO=$LOGS/repro
if [ -f results/baselines/eraft_reproduction.json ]; then
  echo "  present"
else
  if [ ! -f "$REPRO/reference.json" ]; then
    $ERAFT_PY scripts/eraft_reference_run.py --eraft external/E-RAFT \
      --data "$(paths "d['eraft_mvsec_preprocessed']")" \
      --checkpoint "$(paths "d['checkpoints']['eraft_mvsec20']")" --out "$REPRO" \
      > "$LOGS/repro_reference.log" 2>&1
  fi
  $PY scripts/eraft_reproduce.py --reference "$REPRO" > "$LOGS/repro_parity.log" 2>&1 \
    || echo "  reproduction gate NOT met (see results/baselines/eraft_reproduction.json)"
  tail -n 1 "$LOGS/repro_parity.log"
fi

echo "[$(date -Is)] stage 2: M0b selection on mvsec_tune"
if [ -f results/baselines/cm_tune.json ]; then
  echo "  present"
else
  N=$($PY -c "import yaml,math; g=yaml.safe_load(open('configs/cm_baseline.yaml'))['grid']; print(math.prod(len(v) for v in g.values()))")
  seq 0 $((N - 1)) | xargs -P "$CONCURRENCY" -I{} bash -c \
    "$PY scripts/tune_cm_baseline.py --only-cell {} > $LOGS/cm_tune_cell{}.log 2>&1 || echo '  cell {} FAILED'"
  $PY scripts/tune_cm_baseline.py > "$LOGS/cm_tune.log" 2>&1
  tail -n 1 "$LOGS/cm_tune.log"
fi

echo "[$(date -Is)] stage 3: evaluation units, $CONCURRENCY at a time"
units() {
  for m in cm eraft zero; do
    for s in indoor_flying3 indoor_flying2 indoor_flying1 outdoor_day1; do
      for dt in 4 1; do echo "$m mvsec $s $dt"; done
    done
    for s in zurich_city_11_a zurich_city_02_a thun_00_a; do echo "$m dsec $s -"; done
  done
}
run_unit() {   # model dataset sequence dt
  local name="$1_$2_$3_$4" args=(--model "$1" --dataset "$2" --sequence "$3")
  [ "$4" != "-" ] && args+=(--dt "$4")
  if $PY scripts/run_baselines.py "${args[@]}" > "$LOGS/unit_$name.log" 2>&1; then
    echo "[$(date -Is)] $(tail -n 1 "$LOGS/unit_$name.log")"
  else
    echo "[$(date -Is)] $name FAILED (see $LOGS/unit_$name.log)"
  fi
}
export -f run_unit
export PY LOGS
units | xargs -P "$CONCURRENCY" -L 1 bash -c 'run_unit "$@"' _

echo "[$(date -Is)] stage 4: summarise"
$PY scripts/summarise_baselines.py
echo "[$(date -Is)] done"
