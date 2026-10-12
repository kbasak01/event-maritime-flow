# CLAUDE.md — event-maritime-flow

Event-based optical flow for maritime / deck-landing perception, trained on events synthesised with
v2e, evaluated against real event data. Read `IMPLEMENTATION_PLAN.md` §0 before any work; it defines
the rules below. Current phase is recorded in `docs/STATUS.md` — update it at every phase gate.

## Non-negotiables

1. **Evidence tiers.** Every number in `results/`, docs or README carries a tier tag: `T1` (real
   events, real GT), `T2` (v2e from the same sensor's frames, real GT), `T3a` (maritime, RAFT
   teacher → "teacher-EPE", never "EPE"/"accuracy"), `T3b` (maritime, FWL/RSAT only). T3 rows never
   share a table section with T1/T2 rows.
2. **No HDR / latency claims from v2e events.** v2e inherits the source video's dynamic range, blur
   and frame-rate limits. Use the `claim-trace` skill before writing any README or docs prose.
3. **Numbers come from JSON.** Every reported number is produced by a script into `results/*.json`
   and rendered by `scripts/make_tables.py`. Never type a metric value by hand. Files under
   `results/` are written by scripts only (a hook blocks Edit/Write there).
4. **Splits are sacred.** Calibration, CT range, epoch/model selection: `mvsec_tune` only. Test
   sequences are evaluation-only. Fidelity statistics on test sequences are reported, never used for
   a choice.
5. **Long GPU jobs: prepare, then run.** v2e conversions and training runs: prepare the script and
   manifest with the `gpu-job` skill first, then Claude may launch them itself (user authorisation,
   D-006, 2026-10-08). Never point a training or calibration job at a test split.
6. **Licences.** `src/` is MIT. Never copy or import code from `tub-rip/event_based_optical_flow`
   (GPL-3.0). MIT code (E-RAFT, tudelft/event_flow, v2e) may be vendored with its licence header and
   an entry in `docs/LICENSES.md`. No dataset files, frames or event files in git or releases.
7. **Conventions** live in the `event-conventions` skill (= `docs/CONVENTIONS.md`): event store
   layout, polarity {0,1}, time in int64 µs, forward flow in pixels over the window, mask rules.
   Load it before touching `io/`, `events/` or `flow/`.

## Layout (see REPO_SETUP.md for the full tree)

- `src/evflow/io/` readers + canonical store · `events/` slicing, representations · `flow/` metrics,
  IWE, FWL (each with a `*_ref.py` numpy twin) · `sim/` v2e runner, calibration, event statistics ·
  `models/` · `losses/` · `train/` · `eval/` · `teacher/` RAFT · `maritime/` clutter study ·
  `export/` ONNX + benchmark.
- `configs/` YAML only; paths from `configs/paths.local.yaml` (gitignored).
- `splits/` manifests with SHA-256. `results/` JSON (committed) + npz (ignored). `docs/` decisions,
  limitations, gates.

## Commands

- `make lint` (ruff) · `make test` (CPU, no data) · `make test-data` (needs datasets, marker
  `dataset`) · `make smoke` · `make train-smoke` · `make reproduce-tables`.
- Main env `evflow` = uv venv `.venv/` (`source .venv/bin/activate`); v2e runs only in the isolated
  uv venv `external/venvs/v2e/` via `src/evflow/sim/v2e_runner.py` (D-005).
- ONNX Runtime GPU (parity, latency) runs only in the isolated uv venv `external/venvs/ort/`
  (CUDA-13 ORT + TensorRT, no torch; D-063); `PYTHONPATH=src`, `evflow.export.bench` is torch-free.

## Working style

- Before editing: state which phase and which plan step the change serves.
- Tests first for anything in `flow/`, `events/`, `io/store.py`.
- Record every judgement call (thresholds, windows, cuts to corpus size) in `docs/DECISIONS.md` with
  date, reason and the evidence file.
- Agents: `event-data-auditor`, `v2e-fidelity-auditor`, `flow-metric-verifier`,
  `training-run-diagnostician`, `sim2real-study-runner`, `onnx-parity-auditor`,
  `eval-reproducibility-auditor`, `repo-publication-reviewer`. The `phase-gate` skill says which
  ones each gate needs.
- Verdicts from auditors are written to `docs/gates/phase-N.md` verbatim.
