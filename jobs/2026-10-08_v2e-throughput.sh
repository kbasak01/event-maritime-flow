#!/usr/bin/env bash
# Phase 2 step 2 (IMPLEMENTATION_PLAN.md): "Throughput: convert 5 s of MVSEC APS (346x260), 5 s of
# SMD on-board (at 640x360 and 346x260), and 20 LaRS clips; record seconds of compute per second of
# video and per clip to results/v2e_throughput.json. Use this to size the corpus."
# Also sweeps the --timestamp_resolution cap and auto/fixed upsampling and SuperSloMo batch size
# for the MVSEC profile (§1.7: cap chosen in Phase 2). Resumable: finished units are skipped.
#   nohup bash jobs/2026-10-08_v2e-throughput.sh > runs/v2e-throughput/job.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=data/converted/v2e/throughput
LOGS=runs/v2e-throughput
CACHE=data/converted/aps_cache
SEED=1337
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT" "$LOGS"

unit() {  # unit <name> <runner args...>; a failing unit is logged and the job continues
  local name=$1; shift
  echo "[$(date -Is)] unit $name"
  if ! $PY -m evflow.sim.v2e_runner "$@" --unit "$name" --out "$OUT" --seed $SEED \
        --allow-undersampling > "$LOGS/$name.log" 2>&1; then
    echo "[$(date -Is)] unit $name FAILED (see $LOGS/$name.log)"
  fi
  tail -1 "$LOGS/$name.log"
}

# MVSEC APS: first 5 s of mvsec_tune (APS 3651..3870, 220 frames), cap sweep at cutoff_hz 60.
M=(mvsec --sequence outdoor_day2 --aps 3651 3870 --frames-cache "$CACHE" --profile mvsec_grid)
unit mvsec_auto_ts1000      "${M[@]}" --set=--timestamp_resolution=0.001
unit mvsec_auto_ts500       "${M[@]}" --set=--timestamp_resolution=0.0005
unit mvsec_auto_ts250       "${M[@]}" --set=--timestamp_resolution=0.00025
unit mvsec_fixed_ts500      "${M[@]}" --set=--timestamp_resolution=0.0005 \
                                      --set=--auto_timestamp_resolution=False
unit mvsec_auto_ts500_b32   "${M[@]}" --set=--timestamp_resolution=0.0005 --set=--batch_size=32

# SMD on-board: 5 s of the first smd_train video, both output sizes, cap 0.5 ms. Fed as a lossless
# PNG frame folder (the first run of this job passed the AVI directly and v2e crashed, D-025).
for size in 640x360 346x260; do
  unit "smd_${size}_ts500" smd --video MVI_0788_VIS_OB.avi --start 0 --stop 5 \
       --frames-cache data/converted/frames_cache \
       --profile "maritime_${size}" --set=--timestamp_resolution=0.0005
done

# LaRS: 20 lars_train clips (every len//20-th clip of splits/lars_train.json), 640x360, cap 0.5 ms.
mapfile -t CLIPS < <($PY -c "
import json; c = json.load(open('splits/lars_train.json'))['clips']; s = len(c) // 20
print('\n'.join(f\"{x['prefix']}:{x['first']}:{x['last']}\" for x in c[::s][:20]))")
i=0
for clip in "${CLIPS[@]}"; do
  unit "lars_640x360_ts500_$(printf %02d $i)" lars --subset train --clip "$clip" \
       --profile maritime_640x360 --set=--timestamp_resolution=0.0005
  i=$((i + 1))
done

$PY scripts/summarise_throughput.py --units "$OUT" --logs "$LOGS"
echo "[$(date -Is)] done"
