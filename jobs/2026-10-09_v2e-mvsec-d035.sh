#!/usr/bin/env bash
# Phase 3 step 1, re-run after D-035 (timing-constrained mvsec_cal / mvsec_rand frozen from
# results/v2e_calibration_timing.json; first run: jobs/2026-10-09_v2e-mvsec.sh, lagged, D-032).
# IMPLEMENTATION_PLAN.md: "/gpu-job v2e-mvsec: v2e(APS) for mvsec_train,
# mvsec_tune, mvsec_test with mvsec_cal; for v2e_rand_sup, K = 4 additional train corpora with CT
# drawn from the range (seeded). The randomisation is per clip (60 s chunks), not per window."
# Units: src/evflow/sim/corpus.py from configs/v2e_corpus_mvsec.yaml (cuts D-026, seeds and
# chunking D-032). Every APS range is checked against its split manifest by the runner (--split).
# Also converts the real test stores the fidelity report needs (CPU, Phase 1 converter).
# Resumable: existing stores and finished units (store + manifest) are skipped.
# Launch from a clean commit (D-031), e.g. a detached worktree with paths.local.yaml copied in:
#   nohup bash jobs/2026-10-09_v2e-mvsec-d035.sh > runs/v2e-mvsec-d035/job.log 2>&1 &
#   TRIALS_ONLY=1 bash jobs/2026-10-09_v2e-mvsec-d035.sh      # the two <= 5 s trials, then stop
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
export PYTHONPATH=$PWD/src          # this checkout's code, not the editable install's
LOGS=${LOGS:-$MAIN/runs/v2e-mvsec-d035}
CONCURRENCY=${CONCURRENCY:-4}       # D-025: 4-way = 0.48 x sequential
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"

echo "[$(date -Is)] real test stores (CPU)"
EVENTS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['events'])")
for seq in indoor_flying1 indoor_flying2 indoor_flying3 outdoor_day1 outdoor_day2; do
  if [ -f "$EVENTS/mvsec_${seq}_left.h5" ] && [ -f "$EVENTS/mvsec_${seq}_left.json" ]; then
    echo "  $seq: present"
  else
    $PY scripts/convert_events.py mvsec --sequence "$seq" > "$LOGS/real_$seq.log" 2>&1
    echo "  $seq: $(tail -1 "$LOGS/real_$seq.log")"
  fi
done

echo "[$(date -Is)] plan + APS export"
$PY -m evflow.sim.corpus plan
$PY -m evflow.sim.corpus prepare

run_unit() {   # $1 = <corpus>/<name>
  local log="$LOGS/${1//\//_}.log"
  if $PY -m evflow.sim.corpus unit --key "$1" > "$log" 2>&1; then
    echo "[$(date -Is)] $1 $(tail -1 "$log")"
  else
    echo "[$(date -Is)] $1 FAILED (see $log)"
  fi
}
export -f run_unit
export PY LOGS

echo "[$(date -Is)] trials (sequential)"
for k in $($PY -m evflow.sim.corpus units --corpus trial); do run_unit "$k"; done
if [ "${TRIALS_ONLY:-0}" = 1 ]; then echo "[$(date -Is)] trials only: stop"; exit 0; fi

echo "[$(date -Is)] corpus units, $CONCURRENCY at a time"
$PY -m evflow.sim.corpus units --exclude trial | xargs -P "$CONCURRENCY" -I{} bash -c 'run_unit {}'

echo "[$(date -Is)] summarise"
$PY -m evflow.sim.corpus summarise
echo "[$(date -Is)] done"
