#!/usr/bin/env bash
# Phase 2 step 2, follow-up to jobs/2026-10-08_v2e-throughput.sh (run after it, GPU otherwise idle).
# (a) cap 0.75 ms: the coarsest --timestamp_resolution cap that passes v2e's lowpass check at
#     cutoff_hz 60 (eps = dt/tau = 0.737 ms / 2.653 ms = 0.28 < 0.3, v2ecore check_lowpass);
# (b) concurrency: N identical units launched together on the one GPU, to measure the effective
#     GPU time per video second when units overlap (SuperSloMo PNG I/O leaves the GPU ~40 % idle).
# Resumable: finished units are skipped.
#   nohup bash jobs/2026-10-08_v2e-throughput-concurrency.sh > runs/v2e-throughput/job_conc.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=data/converted/v2e/throughput
LOGS=runs/v2e-throughput
CACHE=data/converted/aps_cache
SEED=1337
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT" "$LOGS"
M=(mvsec --sequence outdoor_day2 --aps 3651 3870 --frames-cache "$CACHE" --profile mvsec_grid
   --set=--timestamp_resolution=0.00075 --out "$OUT" --seed $SEED)

$PY -m evflow.sim.v2e_runner "${M[@]}" --unit mvsec_auto_ts750 > "$LOGS/mvsec_auto_ts750.log" 2>&1
tail -1 "$LOGS/mvsec_auto_ts750.log"

# (c) SMD re-run via a lossless PNG frame folder: v2e's video path mis-detects the 1080p colour
#     AVI as monochrome and feeds RGB to SuperSloMo (6-channel crash, D-025).
for size in 640x360 346x260; do
  for ts in 500 750; do
    u="smd_${size}_ts${ts}"
    $PY -m evflow.sim.v2e_runner smd --video MVI_0788_VIS_OB.avi --start 0 --stop 5 \
        --frames-cache data/converted/frames_cache --profile "maritime_${size}" \
        --set=--timestamp_resolution=0.000$ts --unit "$u" --out "$OUT" --seed $SEED \
        > "$LOGS/$u.log" 2>&1 || echo "unit $u FAILED"
    tail -1 "$LOGS/$u.log"
  done
done

for n in 2 3 4; do
  [ -f "$OUT/concurrency_${n}.batch.json" ] && continue
  echo "[$(date -Is)] concurrency $n"
  t0=$(date +%s.%N)
  pids=()
  for i in $(seq 0 $((n - 1))); do
    u="mvsec_auto_ts750_par${n}_${i}"
    $PY -m evflow.sim.v2e_runner "${M[@]}" --unit "$u" > "$LOGS/$u.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  t1=$(date +%s.%N)
  echo "{\"concurrency\": $n, \"batch_wall_s\": $(echo "$t1 - $t0" | bc)}" \
    > "$OUT/concurrency_${n}.batch.json"
  cat "$OUT/concurrency_${n}.batch.json"
done

$PY scripts/summarise_throughput.py --units "$OUT" --logs "$LOGS"
echo "[$(date -Is)] done"
