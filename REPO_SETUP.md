# REPO_SETUP — event-maritime-flow

Workstation: Linux, one RTX A4000 (16 GB), CUDA 12.x driver. Budget ≈ 400 GB free disk for the full
plan (MVSEC `outdoor_day2` alone is large; v2e outputs are compressed with blosc).

## 1. Repository tree (target state)

```
event-maritime-flow/
├── CLAUDE.md  IMPLEMENTATION_PLAN.md  REPO_SETUP.md  SOURCES.md  VALIDATION_CHECKLIST.md  KICKSTART.md
├── README.md                      # written in Phase 10
├── LICENSE                        # MIT
├── pyproject.toml  Makefile  .gitignore
├── .github/workflows/ci.yml       # CPU only, no datasets
├── .claude/
│   ├── settings.json              # permissions + hook
│   ├── hooks/guard_paths.sh       # blocks hand-edits of results/, data/, external/
│   ├── agents/*.md                # 8 subagents
│   └── skills/<name>/SKILL.md     # phase-gate, gpu-job, v2e-convert, claim-trace, event-conventions
├── configs/
│   ├── evflow.yaml                # global defaults (seeds, windows, metrics)
│   ├── v2e_profiles.yaml          # frozen v2e argument profiles
│   ├── eval_mvsec.yaml            # eval windows, crop & mask rules (Phase 1)
│   ├── train_<arm>.yaml           # one per training arm (Phase 5)
│   └── paths.local.yaml           # gitignored; copy from paths.local.yaml.example
├── splits/                        # *.json manifests + SHA256 (committed)
├── src/evflow/
│   ├── io/        store.py mvsec.py dsec.py v2e_reader.py lars.py smd.py
│   ├── events/    slicing.py representations.py representations_ref.py filters.py
│   ├── flow/      metrics.py metrics_ref.py iwe.py iwe_ref.py fwl.py png.py viz.py
│   ├── sim/       v2e_runner.py calibrate.py stats.py
│   ├── models/    evflownet.py cm_baseline.py eraft_ref/ (vendored MIT, licence header)
│   ├── losses/    supervised.py cm.py
│   ├── train/     loop.py
│   ├── eval/      run.py bootstrap.py
│   ├── teacher/   raft.py
│   ├── maritime/  clutter.py
│   └── export/    onnx.py bench.py
├── scripts/       check_data.py make_splits.py fidelity_report.py run_eval.py make_tables.py make_figures.py
├── tests/         test_*.py  fixtures/  oracles/ (vendored MIT metric code; never imported by src)
├── results/       *.json (committed)  *.npz (ignored)  fidelity/ eval/ baselines/
├── docs/          STATUS.md DECISIONS.md CONVENTIONS.md LICENSES.md LIMITATIONS.md gates/
├── data/          (gitignored) dataset files fetched by scripts/fetch_data.py (D-011)
└── external/      (gitignored) v2e checkout, E-RAFT checkout for the reproduction gate
```

## 2. Environments

uv venvs replace the original conda envs (D-005: conda is not installed on the workstation). Both
venvs are gitignored.

### 2.1 Main env `evflow` → `.venv/`

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your driver
uv pip install numpy scipy numba h5py hdf5plugin opencv-python-headless imageio pyyaml tqdm \
               matplotlib onnx onnxruntime-gpu pytest ruff tensorboard
uv pip install -e . --no-deps
python -c "import torch, h5py, hdf5plugin, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
uv pip freeze > requirements.lock
```

`make` targets assume the venv is active. TensorRT EP is optional (install TensorRT matching ORT's
documented compatibility; record the versions).

### 2.2 v2e env → `external/venvs/v2e/` (isolated on purpose)

```bash
git clone https://github.com/SensorsINI/v2e external/v2e
git -C external/v2e rev-parse HEAD > configs/v2e_commit.txt
uv venv -p 3.10 external/venvs/v2e
# CUDA PyTorch first, then v2e (v2e README order)
uv pip install -p external/venvs/v2e torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -p external/venvs/v2e -e external/v2e gdown
# SuperSloMo39.ckpt (Google Drive id from v2e README) → external/v2e/input/SuperSloMo39.ckpt
external/venvs/v2e/bin/gdown 1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx -O external/v2e/input/SuperSloMo39.ckpt
external/venvs/v2e/bin/python external/v2e/v2e.py -h | head -5
uv pip freeze -p external/venvs/v2e > configs/v2e_requirements.lock
```

`sim/v2e_runner.py` calls `<v2e.python> <v2e.repo>/v2e.py …` (both paths from
`paths.local.yaml`). Never import v2e into `evflow`.

### 2.3 Optional oracle envs (outputs only cross over)

- `eraft` (E-RAFT's own `environment.yml`, Python 3.7) for the Phase 4 reproduction gate.
- `oracle-gpl` for `tub-rip/event_based_optical_flow` (GPL-3.0) **only** if you want an independent
  contrast-maximisation reference; never vendor or import it.

## 3. Data acquisition (download only what the plan uses)

| Dataset | What to fetch | Where from | Notes |
|---|---|---|---|
| MVSEC | `outdoor_day1`, `outdoor_day2`, `indoor_flying1–3` `*_data.hdf5`; GT flow `*_gt_flow_dist.npz`; calibration zips | HDF5 + GT-flow Google Drive folders linked from the MVSEC Download page | CC BY-SA 4.0. HDF5 mirrors ROS bag topics (`davis/left/events`, `davis/left/image_raw`, `image_raw_ts`, `image_raw_event_inds`). |
| E-RAFT MVSEC preprocessed | `mvsec_outdoor_day_1_20Hz.tar` | E-RAFT README | Phase 4 reproduction gate only |
| E-RAFT checkpoints | `dsec.tar`, `mvsec_20.tar` | E-RAFT README | MIT |
| DSEC-Flow | 2–3 train sequences with flow GT: `events_left` (events.h5 + rectify_map.h5), `image_timestamps.txt`, flow PNGs + timestamps | DSEC download page | data CC BY-SA 4.0, code MIT (see `docs/LICENSES.md`); 640×480 events, flow at 10 Hz |
| SMD | Visible On-Board (11 videos, 768 MB) | SMD page (Google Drive) | 1920×1080; no licence stated → no redistribution |
| LaRS | Images v1.0.0 (966 MB), Annotations v1.0.0 (22.3 MB), Sequence images v1.0.0 (13.6 GB) | LaRS page (form + terms) | read terms before download; test annotations not public |

`scripts/fetch_data.py` downloads, extracts and sanity-checks everything listed in
`configs/data_sources.yaml` into `data/` (resumable `curl`; Google Drive via its direct `drive.usercontent` URL;
LaRS zips are placed by hand in `data/lars/_archives/` after accepting its terms, then extracted by
the same script). DSEC sequences: `thun_00_a`, `zurich_city_02_a`, `zurich_city_11_a` (D-012). MVSEC
`*_gt.hdf5`/`_odom` are not fetched (D-013).

```bash
python -I scripts/fetch_data.py                 # or --only mvsec,dsec  /  --kind http
python scripts/check_data.py                    # → results/data_inventory.json
```

`scripts/check_data.py` verifies presence, sizes and records SHA-256 of small files.

## 4. `pyproject.toml` (starter)

```toml
[project]
name = "evflow"
version = "0.1.0"
requires-python = ">=3.10"
license = {text = "MIT"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "NPY"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "gpu: needs CUDA",
  "dataset: needs real datasets from paths.local.yaml",
  "slow: > 60 s",
]
addopts = "-m 'not gpu and not dataset and not slow'"
```

## 5. `Makefile` (starter)

```make
PY ?= python
.PHONY: lint test test-data smoke train-smoke reproduce-tables
lint:            ; ruff check src tests scripts
test:            ; pytest -q
test-data:       ; pytest -q -m "dataset" -p no:cacheprovider
smoke:           ; $(PY) -c "import evflow; print('ok')" && pytest -q tests/test_smoke.py
train-smoke:     ; $(PY) -m evflow.train.loop --config configs/train_real_sup.yaml --smoke
reproduce-tables:; $(PY) scripts/make_tables.py --check
```

## 6. `.gitignore` (starter)

```
data/
external/
dumps/
runs/
*.h5
*.hdf5
*.npz
*.onnx
*.pt
*.engine
configs/paths.local.yaml
__pycache__/
.venv/
```

## 7. CI (`.github/workflows/ci.yml`)

CPU-only: checkout, set up Python 3.11, install CPU torch + the main deps, `ruff check`,
`pytest -q` (markers exclude gpu/dataset/slow). Add `tests/test_no_gpl_imports.py` (greps `src/` for
any import of the GPL package) and `tests/test_results_have_tiers.py` (every `results/**/*.json`
summary has a `tier` and `arm` field).

## 8. Claude Code setup

- The `.claude/` folder in this kit goes into the repo root and is committed.
- Make the hook executable: `chmod +x .claude/hooks/guard_paths.sh` (needs `jq`).
- Accept the workspace-trust prompt the first time you start `claude` in the repo; project-level
  hooks only run in trusted folders.
- Start sessions in the repo root so `.claude/agents` and `.claude/skills` load.
