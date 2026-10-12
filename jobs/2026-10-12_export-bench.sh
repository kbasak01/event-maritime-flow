#!/usr/bin/env bash
# Phase 8 steps 1-3 (IMPLEMENTATION_PLAN.md): "Export M1 (best arm) to ONNX opset 17, fp32 and
# fp16; the event-representation build stays outside the graph ... Parity on 200 fixed
# mvsec_test windows: max |Δflow| ≤ 1e-3 px fp32 (TF32 off), ≤ 5e-2 px fp16; EPE change ≤ 0.01
# px. Latency (§1.6) for representation build at 10⁴/10⁵/10⁶ events, network at 256×256, 346×260
# (padded to /16), 640×480; PyTorch eager, ORT CUDA EP, ORT TensorRT EP fp16."
# Runs and windows: configs/evflow.yaml `export`, `latency` (D-063). Torch steps run in the main
# venv; ORT steps in the isolated venv external/venvs/ort (CUDA-13 ORT + TensorRT, no torch;
# D-063). Frozen checkpoints, evaluation-only windows; nothing is chosen. TF32 off throughout.
# Sequential, one CUDA process at a time (D-048b). Resumable: every unit skips existing outputs.
#   nohup bash jobs/2026-10-12_export-bench.sh > runs/export-bench/job.log 2>&1 &
#   SMOKE=1 bash jobs/2026-10-12_export-bench.sh   # 2 windows/sequence, one shape -> runs/export-bench/smoke
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${EVFLOW_PY:-$MAIN/.venv/bin/python}
ORT_PY=${ORT_PY:-$MAIN/external/venvs/ort/bin/python}
export PYTHONPATH=$PWD/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # D-052
LOGS=${LOGS:-$MAIN/runs/export-bench}
RUNS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/paths.local.yaml'))['outputs']['runs'])")
mkdir -p "$LOGS"
echo "[$(date -Is)] code $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs jobs splits | wc -l)"

unit() {  # unit <name> <skip-if-exists path or -> <cmd...>
  local name=$1 done_file=$2
  shift 2
  if [ "$done_file" != - ] && [ -e "$done_file" ]; then
    echo "[$(date -Is)] skip $name ($done_file exists)"
    return 0
  fi
  if "$@" > "$LOGS/$name.log" 2>&1; then
    echo "[$(date -Is)] OK $name: $(grep -v -E '\[W:|\[E:|Warning' "$LOGS/$name.log" | tail -1 | cut -c1-200)"
  else
    echo "[$(date -Is)] FAILED $name (see $LOGS/$name.log)"
    return 1
  fi
}

if [ "${SMOKE:-0}" = 1 ]; then
  S=$LOGS/smoke
  LOGS=$S/logs
  mkdir -p "$LOGS"
  unit export - $PY scripts/export_onnx.py --smoke --out-dir "$S/onnx"
  unit parity_ref "$S/parity/ref.json" $PY scripts/onnx_parity.py ref --work-dir "$S/parity" --windows-per-sequence 2
  unit parity_ort - $ORT_PY scripts/onnx_parity.py ort --work-dir "$S/parity" --onnx-dir "$S/onnx"
  unit parity_torch_fp16 - $PY scripts/onnx_parity.py torch_fp16 --work-dir "$S/parity"
  unit parity_summary - $PY scripts/onnx_parity.py summarise --work-dir "$S/parity" --out "$S/onnx_parity.json"
  unit latency_torch "$S/latency/torch.json" $PY scripts/latency_bench.py torch --smoke --work-dir "$S/latency"
  unit latency_ort "$S/latency/ort.json" $ORT_PY scripts/latency_bench.py ort --smoke --work-dir "$S/latency" --onnx-dir "$S/onnx"
  unit latency_summary - $PY scripts/latency_bench.py summarise --work-dir "$S/latency" --out "$S/latency.json"
  echo "[$(date -Is)] smoke done"
  exit 0
fi

unit export - $PY scripts/export_onnx.py
unit parity_ref "$RUNS/export/parity/ref.json" $PY scripts/onnx_parity.py ref
unit parity_ort - $ORT_PY scripts/onnx_parity.py ort
unit parity_torch_fp16 - $PY scripts/onnx_parity.py torch_fp16   # diagnostic, not gated (auditor F-2)
unit latency_torch "$RUNS/export/latency/torch.json" $PY scripts/latency_bench.py torch
unit latency_ort "$RUNS/export/latency/ort.json" $ORT_PY scripts/latency_bench.py ort
echo "[$(date -Is)] summarise"
unit parity_summary - $PY scripts/onnx_parity.py summarise
unit latency_summary - $PY scripts/latency_bench.py summarise
unit tables - $PY scripts/make_tables.py
echo "[$(date -Is)] done"
