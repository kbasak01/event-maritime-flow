#!/usr/bin/env bash
# Phase 6: evaluate every trained M1 run (+ T2 zero flow), then summarise, tables, figures.
# IMPLEMENTATION_PLAN.md Phase 6:
#   1. "scripts/run_eval.py -> results/eval/<arm>/<seed>/<dataset>_<seq>_dt<k>.json with
#      per-window metric arrays (.npz, gitignored) and summary JSON (committed)."
#   2. "scripts/make_tables.py -> results/TABLES.md" (Tables A-D)   3. Figures.
# Matrix (src/evflow/eval/study.py, D-051; T2 scope = user decision 2026-10-11):
#   T1 mvsec_test (real events), dt 1 and 4 ... all 17 runs of results/train/summary.json
#   T1 dsec_test (real events, 100 ms) ......... the 11 voxel5 runs
#   T2 mvsec_test (v2e(APS) events, D-035) ..... the 11 voxel5 runs + M0a (zero flow)
#   repro: real_sup s1337 outdoor_day1 dt1 re-evaluated into runs/eval/repro/ (determinism check
#          against the Phase 5 JSON, summarise_eval.py)
# Test splits are read through frozen checkpoints only (best.pt selected on mvsec_tune, SHA-256
# verified by run_eval.py); nothing is trained, tuned or selected here (CLAUDE.md rule 4).
# One CUDA process at a time (D-048b): units run sequentially. Each unit skips if its JSON exists
# (resumable); a failed unit is reported and does not stop the job. Unit logs are appended to.
# Launch from a clean commit in the main tree (code must stay unchanged while it runs):
#   nohup bash jobs/2026-10-11_eval-all.sh > runs/eval-all/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-11_eval-all.sh   # 20 windows of one T1, T2, DSEC and M0a-T2 unit
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${EVFLOW_PY:-$PWD/.venv/bin/python}
export PYTHONPATH=$PWD/src
LOGS=${LOGS:-$PWD/runs/eval-all}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# D-052: torch's default 18 OpenMP threads make per-window CPU metrics ~40x slower on this WSL2
# host (attempt 1, runs/eval-all/job.attempt1.log); 4 threads: ~37 windows/s.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p "$LOGS"
echo "[$(date -Is)] threads $OMP_NUM_THREADS code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"
RUNS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['runs'])")/train

m1() {      # run_id log-name run_eval-args...
  local run="$1" name="$2"; shift 2
  if $PY scripts/run_eval.py --run-dir "$RUNS/$run" "$@" >> "$LOGS/$name.log" 2>&1; then
    echo "[$(date -Is)] $(tail -n 1 "$LOGS/$name.log")"
  else
    echo "[$(date -Is)] $name FAILED (see $LOGS/$name.log)"
  fi
}
zero_v2e() {   # sequence dt
  local name="zero_flow_mvsec_$1_dt$2_v2e"
  if $PY scripts/run_baselines.py --model zero --dataset mvsec --sequence "$1" --dt "$2" \
       --events v2e >> "$LOGS/$name.log" 2>&1; then
    echo "[$(date -Is)] $(tail -n 1 "$LOGS/$name.log")"
  else
    echo "[$(date -Is)] $name FAILED (see $LOGS/$name.log)"
  fi
}

if [ "${SMOKE:-0}" = 1 ]; then
  R=real_sup_voxel5_s1337
  m1 $R smoke_t1 --dataset mvsec --sequence indoor_flying1 --dt 1 --max-windows 20 --force
  m1 $R smoke_t2 --dataset mvsec --sequence indoor_flying2 --dt 4 --events v2e --max-windows 20 --force
  m1 $R smoke_dsec --dataset dsec --sequence zurich_city_02_a --max-windows 20 --force
  $PY scripts/run_baselines.py --model zero --dataset mvsec --sequence outdoor_day1 --dt 1 \
    --events v2e --max-windows 20 --force > "$LOGS/smoke_zero_v2e.log" 2>&1
  echo "[$(date -Is)] $(tail -n 1 "$LOGS/smoke_zero_v2e.log")"
  echo "[$(date -Is)] smoke done (outputs under runs/eval/smoke, runs/phase4-baselines/smoke)"
  exit 0
fi

# run ids, voxel5 main runs first (Table A/B), then the ablation (Table C)
mapfile -t ALL < <($PY - <<'EOF'
import sys
sys.path.insert(0, "src")
from evflow.eval import study
runs = study.trained_runs()
runs.sort(key=lambda r: (r["representation"] != study.MAIN_REP, r["run_id"]))
for r in runs:
    print(r["run_id"], r["representation"])
EOF
)
SEQS="indoor_flying1 indoor_flying2 indoor_flying3 outdoor_day1"
DSEC="thun_00_a zurich_city_02_a zurich_city_11_a"
echo "[$(date -Is)] ${#ALL[@]} runs"

echo "[$(date -Is)] stage 1: T1 mvsec_test, every run, dt 1 and 4"
for line in "${ALL[@]}"; do
  read -r run rep <<< "$line"
  for seq in $SEQS; do for dt in 1 4; do
    m1 "$run" "${run}_mvsec_${seq}_dt${dt}" --dataset mvsec --sequence "$seq" --dt "$dt"
  done; done
done

echo "[$(date -Is)] stage 2: T2 mvsec_test on v2e(APS) events, voxel5 runs + zero flow"
for seq in $SEQS; do for dt in 1 4; do zero_v2e "$seq" "$dt"; done; done
for line in "${ALL[@]}"; do
  read -r run rep <<< "$line"
  [ "$rep" = voxel5 ] || continue
  for seq in $SEQS; do for dt in 1 4; do
    m1 "$run" "${run}_mvsec_${seq}_dt${dt}_v2e" --dataset mvsec --sequence "$seq" --dt "$dt" \
      --events v2e
  done; done
done

echo "[$(date -Is)] stage 3: T1 dsec_test, voxel5 runs"
for line in "${ALL[@]}"; do
  read -r run rep <<< "$line"
  [ "$rep" = voxel5 ] || continue
  for seq in $DSEC; do m1 "$run" "${run}_dsec_${seq}" --dataset dsec --sequence "$seq"; done
done

echo "[$(date -Is)] stage 4: reproduction unit"
m1 real_sup_voxel5_s1337 repro_real_sup_outdoor_day1_dt1 --dataset mvsec \
  --sequence outdoor_day1 --dt 1 --out-dir runs/eval/repro/real_sup/1337 --force

echo "[$(date -Is)] stage 5: summary, tables, figures"
$PY scripts/summarise_eval.py 2>&1 | tee "$LOGS/summarise.log" | tail -n 1 || true
$PY scripts/make_tables.py 2>&1 | tail -n 1 || true
$PY scripts/make_figures.py > "$LOGS/figures.log" 2>&1 && tail -n 1 "$LOGS/figures.log" \
  || echo "[$(date -Is)] make_figures FAILED (see $LOGS/figures.log)"
echo "[$(date -Is)] done"
