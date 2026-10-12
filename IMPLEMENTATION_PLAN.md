# `event-maritime-flow` — Sequential Implementation Plan

Event-based optical flow for maritime / deck-landing perception with **no event-camera hardware**:
synthesize DVS events from ordinary video with v2e, train compact event-flow networks, and measure
honestly (a) how much of the synthetic→real gap is due to the simulator, using real event data with
real ground truth, and (b) what happens when the same models meet sea-surface video, where no
ground-truth flow exists. Single RTX A4000 (16 GB). Standalone repo (no P1 submodule); borrows P1's
benchmarking methodology and repo conventions.

Plan date: 2026-10-08. Every external fact below was checked on that date and is listed with its
source in `SOURCES.md`; items marked **(verify)** could not be confirmed and are Phase 0 checks.

---

## 0. Read this first: the five things that decide whether this project works

**(1) v2e cannot add information the source video does not contain.** v2e turns frames into events
by modelling log-intensity change, photoreceptor bandwidth, threshold variation and noise, and it
fills the time between frames by SuperSloMo interpolation. It inherits the source camera's 8-bit
dynamic range, its motion blur, its compression artefacts and its frame-rate aliasing. So synthetic
maritime events can demonstrate the *pipeline* and the *motion content*, but they cannot demonstrate
the two reasons anyone wants an event camera on a ship: high dynamic range under sun glare and
sub-millisecond latency. **No sentence in this repo may claim glare/HDR robustness or latency
advantages from v2e-generated events.** v2e also stacks multiple per-frame events into timestamp
"pyramids" between interpolated frames (documented in its README) — a temporal artefact a network
can learn and that real sensors do not have. The finding to report is what survives these limits.

**(2) There is no ground-truth flow for real maritime video, so every number carries an evidence tier.**

| Tier | Events | Flow reference | What it can support | Datasets |
|---|---|---|---|---|
| **T1** | real | real GT (depth + pose motion field, or LiDAR) | accuracy claims | MVSEC, DSEC-Flow |
| **T2** | v2e from frames *of the same sensor* | the same real GT as T1 | simulator-gap claims | MVSEC (APS → v2e) |
| **T3a** | v2e from maritime video | RAFT on the source frames ("teacher") | *agreement with a frame-based teacher*, never accuracy | SMD, LaRS |
| **T3b** | v2e from maritime video | none — FWL / RSAT self-consistency | sharpness of motion compensation, never accuracy | SMD, LaRS |

Every table row and every README number carries its tier tag. T3 numbers never appear in the same
table section as T1/T2 numbers. "EPE" is reserved for T1/T2; T3a reports "teacher-EPE".

**(3) The matched-content experiment is the scientific core, and MVSEC is what makes it possible.**
MVSEC's DAVIS 346B cameras record APS grayscale frames and DVS events through the same lens on the
same 346×260 pixel grid, and MVSEC ships dense ground-truth flow for the left camera. Running v2e on
the APS frames therefore produces synthetic events in exactly the geometry of the real events, and
the **same GT flow supervises both**. Train the identical network on real events vs v2e events with
identical labels, evaluate both on real events: the difference is the simulator gap, isolated from
content, viewpoint and label noise. DSEC cannot do this (its frame cameras are separate sensors with
a different viewpoint), so DSEC is a T1 generalisation test only.

**(4) Contrast threshold is the lever, and it must be calibrated on data the test never sees.**
Stoffregen et al. (ECCV 2020) found the simulated contrast threshold (CT) is the key parameter for
sim-to-real transfer of event CNNs, and that real cameras' apparent CT varies widely even within one
dataset. So: (a) calibrate v2e's `--pos_thres/--neg_thres/--sigma_thres/--cutoff_hz/noise` by
matching event statistics on a **tune slice** of the MVSEC training sequence only; (b) compare a
fixed calibrated CT against train-time CT randomisation; (c) report how much of the gap each closes.
Calibrating on a test sequence would be leakage — the `event-data-auditor` checks this.

**(5) Water breaks the assumptions of optical flow and dominates the event budget.** Sea surface is
non-rigid and specular; sun glint flickers; brightness constancy fails; the "true" flow of a wave
crest is ill-defined. A maritime event-camera study (arXiv 2202.04231, DAVIS346red) reports that in
rough seas waves produce many times more events than the vessels of interest. Consequences: every
maritime metric is sliced by semantic class (water / sky / obstacle, from LaRS keyframe labels);
whole-image averages are reported only alongside the slices; and a **sea-clutter event-budget
characterisation** (what fraction of events comes from water, as a function of v2e CT) is itself a
deliverable — synthetic, labelled as such.

**Positioning (do not over-claim).** Synthetic-event optical-flow datasets exist for underwater scenes
(eStonefish-Scenes 2025; UEOF, WACVW 2026, built with a video-to-event pipeline on ray-traced RGBD),
and the contrast-threshold sim-to-real effect is established (Stoffregen 2020). A search on
2026-10-08 found no above-water maritime event-flow dataset or study. This project's contributions:
(i) a matched-supervision quantification of v2e's sim-to-real gap for event flow on MVSEC, with
calibration vs randomisation; (ii) an above-water maritime v2e pipeline with class-sliced T3
evaluation and a sea-clutter event-budget study; (iii) deployment numbers for the event
representation + network on one GPU. Do not claim novelty for "training flow on synthetic events".

---

## 1. System design (decided up front, so Claude Code isn't improvising)

### 1.1 Architecture

```
 frames (MVSEC APS 346×260 | SMD 1920×1080 | LaRS 10-frame clips)
      │                                                       real events (MVSEC, DSEC)
      ▼                                                               │
 sim/v2e_runner ── separate `v2e` conda env, CLI subprocess,          │
      │            pinned args + seed + manifest SHA                  │
      ▼                                                               ▼
 io/  ───────────── canonical EventStore HDF5 (DSEC-like layout) ◀────┘
      │
 events/slicing ─▶ events/representations (voxel grid | time surface | count image)
      │
      ├─▶ models/evflownet  (M1, U-Net, ours)        ─┐
      ├─▶ models/cm_baseline (M0b, per-window CMax)    ├─▶ flow (px per window, forward)
      └─▶ models/eraft_ref  (R1, MIT, inference only) ─┘
                                                          │
 flow/metrics (EPE, %Out, NPE, AE)  flow/iwe + fwl (FWL, RSAT)  teacher/raft (T3a)
      │
 results/*.json  ──▶ scripts/make_tables.py ──▶ results/TABLES.md, figures
```

Pure-numpy reference implementations live in `flow/metrics_ref.py`, `flow/iwe_ref.py` and
`events/representations_ref.py`; the torch versions used in training are tested equal to them.
`sim/` is the only code that touches v2e, and it only calls the v2e CLI — no v2e import into the
main environment.

### 1.2 Canonical event store (one format for every source)

HDF5, mirroring DSEC so one slicer serves all datasets:

| Key | dtype | Meaning |
|---|---|---|
| `events/x`, `events/y` | uint16 | pixel coordinates, origin top-left, *rectified/undistorted space stated in attrs* |
| `events/t` | int64 | microseconds, monotonic non-decreasing, relative to `t_offset` |
| `events/p` | int8 | polarity in {0 = OFF, 1 = ON} |
| `ms_to_idx` | int64 | `t[ms_to_idx[ms]] >= ms*1000` and `t[ms_to_idx[ms]-1] < ms*1000` (DSEC definition) |
| `t_offset` | int64 | microseconds to add to recover sensor time |
| attrs | — | `width`, `height`, `source`, `sensor`, `distortion` (`raw`/`rectified`), `simulator_manifest_sha256` (synthetic only), `polarity_convention` |

Converters: `io/mvsec.py` (hdf5 `davis/left/events` + APS frames + `*_gt_flow_dist.npz`),
`io/dsec.py` (native layout + `rectify_map.h5` + 16-bit flow PNG), `io/v2e_reader.py` (v2e
`--dvs_h5` output — **inspect its dataset names and column order with `h5ls -v` before writing the
converter; do not assume**), `io/lars.py`, `io/smd.py`. MVSEC's event column order and time units
are likewise verified in Phase 1 against the file, not assumed.

### 1.3 Data splits (fixed, seeded 20261008, manifests with SHA-256 in `splits/`)

| Split | Source | Used for | Labels read? |
|---|---|---|---|
| `mvsec_train` | `outdoor_day2`, all but the tune slice | training every MVSEC arm | yes |
| `mvsec_tune` | `outdoor_day2`, one contiguous 60 s slice, ≥ 30 s gap from train windows | v2e calibration, model/epoch selection, CT-randomisation range | yes |
| `mvsec_test` | `indoor_flying1–3`, `outdoor_day1` (standard eval windows) | T1/T2 evaluation only | eval only |
| `dsec_test` | 2–3 DSEC-Flow *train* sequences never used for training | T1 generalisation, 640×480 | eval only |
| `smd_train` / `smd_test` | SMD visible on-board, split **by video** (8 / 3) | maritime self-supervised training / T3 eval | no GT exists |
| `lars_train` / `lars_eval` | LaRS train / val (test annotations are not public) | maritime training / class-sliced T3 eval | semantic masks for *slicing only* |

The `outdoor_day1` evaluation window is not standardised in the literature (at least two index sets
are in use). Fix one, record its frame indices in `configs/eval_mvsec.yaml`, and state it in the
README. Same for the MVSEC "sparse" mask (pixels with events in the window) and the bottom-row crop
for the car hood: write the exact rule into config, never into code constants.

### 1.4 Models and training arms

| ID | Model | Notes |
|---|---|---|
| M0a | zero flow | FWL = 1 by construction; anchors every table |
| M0b | per-window contrast maximisation (our own implementation, affine-flow or patch-wise) | classical baseline, no learning |
| M1 | EV-FlowNet-style U-Net (≈ 5–15 M params), input = event representation of one window | **the** model of this study; ONNX-friendly |
| R1 | E-RAFT, DSEC checkpoint (MIT), inference only | reference row, tagged `reference_external` |
| M2 *(stretch)* | ConvGRU variant of M1 | only after Phase 9 |

Training arms for M1 (each tagged; every arm is trained on `mvsec_train` windows):

| Arm tag | Training events | Supervision |
|---|---|---|
| `real_sup` | real DVS | MVSEC GT flow |
| `v2e_fixed_sup` | v2e(APS), calibrated fixed CT | **same** GT flow |
| `v2e_rand_sup` | v2e(APS), CT randomised per clip within the tune-derived range | same GT flow |
| `v2e_selfsup` | v2e(APS), calibrated | contrast-maximisation loss, no GT |
| `real_selfsup` | real DVS | contrast-maximisation loss, no GT |
| `maritime_selfsup` | v2e(SMD + LaRS train) | contrast-maximisation loss |
| `maritime_teacher` | v2e(SMD + LaRS train) | RAFT pseudo-GT with forward–backward consistency mask |

Seeds 1337, 2026, 7 for `real_sup`, `v2e_fixed_sup`, `v2e_rand_sup`; one seed for the rest unless
time allows. The contrast-maximisation loss must include a guard against **event collapse** (Shiba et
al., Sensors 2022): a divergence/deformation regulariser or a multi-reference-time IWE; the
`training-run-diagnostician` checks for collapse signatures (flow magnitude blowing up, IWE variance
rising while warped events concentrate on few pixels).

### 1.5 Event representations (ablation on `real_sup` and `v2e_fixed_sup`, one seed)

`voxel` (B = 5 and 15 temporal bins, bilinear in time, signed polarity, normalised over non-zero
cells — E-RAFT's convention), `timesurface` (per-polarity exponential decay, τ = window/3),
`count` (per-polarity counts). The question: is the simulator gap representation-dependent?
(Hypothesis to test, not assume: count images are less sensitive to v2e's timestamp pyramids.)

### 1.6 Metrics (implement exactly; each has a numpy reference)

- **EPE / AEE** (px), **%Out** (EPE > 3 px *and* > 5 % of GT magnitude — MVSEC convention),
  **NPE** for N ∈ {1, 2, 3} (DSEC convention), **AE** angular error (deg). Masks: valid-GT ∧
  (MVSEC sparse: events-in-window) ∧ crop rule. Units: pixels of displacement over the window,
  forward flow from window start to window end.
- **dt = 1 and dt = 4** for MVSEC (window = 1 or 4 APS frame intervals; ≈ 32 ms indoor, ≈ 22 ms
  outdoor for dt = 1). Flow scales with dt; never compare across dt.
- **FWL** (Stoffregen et al. 2020): Var(IWE warped by predicted flow) / Var(IWE with zero flow); > 1
  is better. **RSAT** (Hagenaars et al. 2021). Both are self-supervised; T3b uses only these.
- **teacher-EPE** (T3a): EPE against RAFT pseudo-GT on pixels passing a forward–backward
  consistency check (|f + b∘f| < 0.01(|f|²+|b∘f|²) + 0.5 px; implemented in the standard squared
  form |f + b∘f|² < …, D-055), reported with the mask coverage.
- **Class slices** (maritime): every T3 metric per LaRS class {water, sky, obstacle}, plus all.
- **Event statistics** (simulator fidelity): events/s per window, ON/OFF ratio, per-pixel rate
  distribution (log histogram), inter-event-interval distribution per pixel, spatial
  autocorrelation of count images, fraction of "hot" pixels. Distance real↔synthetic: Wasserstein-1
  on each distribution, reported per sequence.
- **Uncertainty of results**: moving-block bootstrap over evaluation windows (block = 1 s), 1,000
  resamples, 95 % CI on every T1/T2 mean; paired bootstrap for arm-vs-arm differences.
- **Gap metrics**: `gap = EPE(v2e arm) − EPE(real_sup)` on `mvsec_test`;
  `closure = 1 − gap(v2e_rand_sup)/gap(v2e_fixed_sup)`. Report with paired-bootstrap CIs.
- **Latency**: representation build (as a function of events per window: 10⁴, 10⁵, 10⁶) and network
  forward, separately and end-to-end; p50/p90/p99/max; ≥ 50 warm-up, ≥ 500 timed iterations,
  CUDA-synchronised brackets (P1 methodology); PyTorch fp32, ORT CUDA EP fp32/fp16, TensorRT EP fp16.

### 1.7 v2e configuration policy

- Runs in its own conda env (`v2e`, Python 3.10 per v2e's README) from a pinned commit; invoked only
  through `sim/v2e_runner.py`, which writes a manifest (commit, full argv, seed, input SHA-256,
  output SHA-256, wall time, GPU) next to every output.
- Always: `--no_preview --skip_video_output --dvs_emulator_seed <seed> --dvs_h5 <file>`; explicit
  `--input_frame_rate` from `ffprobe` (or the dataset's documented rate) — never v2e's guess.
- MVSEC: `--dvs346` (346×260) on the raw APS frames, `--auto_timestamp_resolution` with a
  `--timestamp_resolution` cap chosen in Phase 2. SMD/LaRS: `--output_width 640 --output_height 360`
  (16:9 preserved) unless Phase 2 throughput forces smaller.
- Trial every new configuration with `--stop_time` on ≤ 5 s before a long run (v2e's own advice;
  its README warns conversion is many times slower than real time and scales with 1/timestamp
  resolution).

### 1.8 Stretch (only after Phase 9 passes)

- Event-based water/sky/obstacle segmentation on v2e(LaRS) with keyframe labels (the "motion
  segmentation" half of the original idea), evaluated per class.
- Simulator ablation: ESIM via `rpg_vid2e` vs v2e on the matched-content MVSEC experiment.
- DSEC-Flow public benchmark submission of the best arm (`check_optical_flow_submission.py`).
- M2 recurrent model; DSEC fine-tuning.

---

## 2. Phase plan

Each phase ends with an exit gate. Run `/phase-gate N` before starting the next phase. Long GPU jobs
(v2e conversion, training) are run by the user from commands Claude prepares with `/gpu-job`.

| Phase | Name | Days | GPU | Exit gate in one line |
|---|---|---:|:-:|---|
| 0 | Scaffold, environments, licences, data access | 0.5 | – | `make smoke` green; both envs import; licence table filled |
| 1 | Event core: store, slicing, representations, metrics, IWE/FWL ⚠️ crux | 1 | – | analytic fixtures pass; metric parity with oracle; auditors PASS |
| 2 | v2e runner, throughput budget, CT calibration | 1 | ✔ | throughput JSON; calibrated profile chosen on `mvsec_tune` only |
| 3 | Matched-content MVSEC corpus + simulator-fidelity report | 1 | ✔ | v2e(APS) for train/tune/test; fidelity JSON; `v2e-fidelity-auditor` PASS |
| 4 | Baselines + reference reproduction gate | 1 | ✔ | M0a/M0b/R1 numbers; R1 within tolerance of its published DSEC/MVSEC figure or miss explained |
| 5 | Train MVSEC arms (+ representation ablation) | 2 | ✔ | all arms trained, selected on `mvsec_tune`; no collapse |
| 6 | Sim-to-real headline (T1/T2) ⭐ | 1 | – | gap/closure table with CIs; DSEC generalisation rows; tier tags clean |
| 7 | Maritime track (T3) + sea-clutter study | 1.5 | ✔ | class-sliced T3 tables; clutter curves; no T3 number in a T1/T2 section |
| 8 | ONNX export, parity, latency | 1 | ✔ | parity + latency JSON committed |
| 9 | Validation sweep | 1 | – | every `VALIDATION_CHECKLIST.md` line has a verdict |
| 10 | Publication | 0.5–1 | – | `repo-publication-reviewer`: SHIP |

**Total ≈ 11.5 working days.** **Fast path (≈ 6 days):** 0 → 1 → 2 (calibration on a 3×3 grid) →
3 (`indoor_flying1` + tune slice + 10 min of `outdoor_day2`) → 5 (`real_sup` and `v2e_fixed_sup`,
one seed, voxel only) → 6 (MVSEC only) → 7-lite (two SMD clips, T3b FWL only, plus the clutter
study on 200 LaRS val keyframes) → 10. That still yields the headline matched-content gap and a
maritime demonstration.

---

### Phase 0 — Scaffold, environments, licences, data access (0.5 day)

1. `git init event-maritime-flow`; copy this starter kit in (`CLAUDE.md`, this plan, `REPO_SETUP.md`,
   `SOURCES.md`, `VALIDATION_CHECKLIST.md`, `KICKSTART.md`, `configs/`, `docs/`, `.claude/`).
2. Environments per `REPO_SETUP.md`: `evflow` (main, Python 3.11, PyTorch 2.x CUDA, onnxruntime-gpu,
   h5py + hdf5plugin, numba, opencv, torchvision for RAFT) and `v2e` (conda, Python 3.10, v2e at a
   pinned commit, SuperSloMo checkpoint in place). The GPL-licensed `tub-rip/event_based_optical_flow`
   is **never** installed into `evflow`; if used at all it lives in a throwaway `oracle-gpl` env and
   only its *outputs* cross over.
3. Package skeleton `src/evflow/` (layout in `REPO_SETUP.md`), `pyproject.toml` (ruff + pytest markers
   `gpu`, `dataset`, `slow`), `Makefile`, CPU CI workflow.
4. `configs/paths.local.yaml` (gitignored) from the example; every dataset path comes from it.
5. **Licence and access check** — fill `docs/LICENSES.md` from `SOURCES.md`, resolving every
   **(verify)** item: DSEC data licence, UZH-FPV licence (if used), LaRS terms (read
   `terms.html` before downloading), SMD terms (none stated on its page → treat as
   no-redistribution). Decide now what imagery may appear in the README.
6. `scripts/check_data.py`: confirms each dataset path exists, records file sizes and SHA-256 of the
   small files (GT npz, flow PNG index, LaRS annotation zip) to `results/data_inventory.json`.

**Exit gate:** `make lint test smoke` green locally and in CI; `python -c "import evflow"` in
`evflow`; `v2e -h` (or `python v2e.py -h`) runs in `v2e`; `docs/LICENSES.md` has no **(verify)** left.

---

### Phase 1 — Event core (1 day) ⚠️ the crux

Everything downstream calls this. A sign error in flow, an off-by-one in `ms_to_idx`, or a polarity
flip produces plausible-looking but wrong numbers no plot will expose.

1. `io/store.py`: write/read the canonical store (§1.2); `EventSlicer` with DSEC's `ms_to_idx`
   semantics (`t_start <= t < t_end`); converters for MVSEC (inspect the file first and record the
   observed column order, units and polarity encoding in `docs/DECISIONS.md`) and DSEC (rectify map
   applied, 16-bit PNG flow decode `(v − 2¹⁵)/128`, valid = channel 3 == 1).
2. `events/representations{,_ref}.py`: voxel, time surface, count — torch (GPU, `index_put_` with
   accumulate) and numpy reference.
3. `flow/metrics{,_ref}.py`: EPE, %Out, NPE, AE with explicit mask arguments.
4. `flow/iwe{,_ref}.py` + `flow/fwl.py`: image of warped events with bilinear splatting to a
   reference time, FWL, RSAT.
5. `docs/CONVENTIONS.md` is the `event-conventions` skill — keep them identical.
6. Tests (no dataset, no GPU):
   - **analytic fixtures**: a bar/edge translating at known (u, v) px/s generates events
     analytically; correct flow → maximal IWE variance (FWL > 1) and zero-flow FWL = 1 exactly;
     wrong-sign flow → FWL < correct-flow FWL;
   - slicer: the DSEC docstring example (`t = [0, 500, 2100, …]` → `ms_to_idx = [0, 2, 2, 3, …]`)
     reproduced exactly; empty windows; windows past the end return `None`;
   - representations: total signed mass conserved for count; voxel bin assignment at bin edges;
     torch == numpy within 1e-6;
   - metrics: hand-computed 4×4 cases; masks honoured; %Out both conditions required;
   - DSEC flow PNG round-trip encode/decode;
   - **oracle parity**: copy the MIT-licensed metric functions of `tudelft/event_flow` into
     `tests/oracles/` (with licence header; never imported by `src/`) and match AEE/%Out/FWL on
     random inputs to 1e-6.

**Exit gate:** all tests green; `flow-metric-verifier` **PASS**; `event-data-auditor` **PASS** on the
converters run against one real MVSEC and one real DSEC file (marked `dataset`).

---

### Phase 2 — v2e runner, throughput budget, CT calibration (1 day, GPU)

1. `sim/v2e_runner.py` + `configs/v2e_profiles.yaml`; manifest per output (§1.7). Converts v2e h5 to
   the canonical store.
2. **Throughput**: convert 5 s of MVSEC APS (346×260), 5 s of SMD on-board (at 640×360 and 346×260),
   and 20 LaRS clips; record seconds of compute per second of video and per clip to
   `results/v2e_throughput.json`. **Use this to size the corpus**: total v2e budget ≤ 12 GPU-hours
   for the full plan; cut `mvsec_train` minutes and SMD minutes to fit, and write the cut into
   `docs/DECISIONS.md`.
3. **Calibration** (`sim/calibrate.py`) on `mvsec_tune` only: grid over `pos_thres = neg_thres ∈
   {0.15, 0.2, 0.3, 0.4}`, `sigma_thres ∈ {0.02, 0.04}`, `cutoff_hz ∈ {0, 60}`, shot noise ∈ {0,
   1 Hz}; objective = mean Wasserstein-1 over the §1.6 event statistics vs real DVS on the same
   slice. Pick the argmin as profile `mvsec_cal`; the CT-randomisation range for `v2e_rand_sup` is the
   set of thresholds within 1.5× of the minimum objective, widened to at least ±30 % around the
   argmin (Stoffregen's observation that real CT varies). Write `results/v2e_calibration.json`.
4. `v2e-fidelity-auditor` on the calibration: tune-only, no test sequence touched.

**Exit gate:** throughput and calibration JSON committed; profile and randomisation range frozen in
`configs/v2e_profiles.yaml` with the calibration JSON's SHA.

---

### Phase 3 — Matched-content MVSEC corpus + simulator-fidelity report (1 day, GPU; v2e overnight)

1. `/gpu-job v2e-mvsec`: v2e(APS) for `mvsec_train`, `mvsec_tune`, `mvsec_test` with `mvsec_cal`;
   for `v2e_rand_sup`, K = 4 additional train corpora with CT drawn from the range (seeded). The
   randomisation is per clip (60 s chunks), not per window — v2e is offline.
2. `scripts/fidelity_report.py`: §1.6 event statistics, real vs v2e, per test sequence (allowed:
   statistics are not labels, but they are **reported only, never used for any choice**) →
   `results/fidelity/*.json` + figures (rate histograms, ON/OFF, per-pixel rate maps side by side).
3. Check alignment: v2e events and real events in the same window should co-locate on edges —
   compute the normalised cross-correlation of count images, real vs synthetic, per window; a low
   value flags a frame-timing or rectification bug before any training happens.

**Exit gate:** corpus manifests complete; cross-correlation median above the threshold set in
Phase 2 on the tune slice; `v2e-fidelity-auditor` **PASS** (provenance, no test-driven choices,
claims about fidelity limited to measured statistics).

---

### Phase 4 — Baselines and reference reproduction gate (1 day, GPU)

1. M0a, M0b on `mvsec_test` (dt = 1, 4) and `dsec_test`.
2. R1 (E-RAFT DSEC checkpoint) on `dsec_test` with its own 15-bin voxel / 100 ms setting. R1 was
   trained on DSEC train sequences, so its `dsec_test` numbers are tagged `reference_seen_data` —
   they validate the pipeline (low EPE expected), not generalisation.
3. **Reproduction gate**: R1's MVSEC 20 Hz checkpoint on `outdoor_day1` using E-RAFT's
   pre-processed MVSEC release; our metric code must reproduce E-RAFT's own evaluation output
   on the same flows to 1e-3 px, and the EPE should be within 10 % of the value in the E-RAFT
   paper. A miss is investigated (mask, crop, window, dt) and documented, not loosened.

**Exit gate:** `results/baselines/*.json`; reproduction gate met or the miss explained in
`docs/DECISIONS.md` and accepted by `flow-metric-verifier`.

---

### Phase 5 — Train the MVSEC arms (2 days, GPU)

1. `models/evflownet.py`, `losses/{supervised,cm}.py`, `train/loop.py` (bf16 autocast, AdamW,
   cosine schedule, random crop 256×256 + horizontal flip with flow sign flip, polarity flip
   augmentation *off* by default), `make train-smoke` (overfit 8 windows: EPE < 0.1 px).
2. `/gpu-job train-<arm>-<seed>` for every arm in §1.4 (MVSEC arms only). Selection on
   `mvsec_tune` EPE (supervised arms) or FWL (self-supervised arms) only; checkpoint SHA-256 recorded.
3. Representation ablation (§1.5).
4. `training-run-diagnostician` on every run: curves, divergence, collapse signatures, selection
   leakage.

**Exit gate:** all planned runs present with manifests; diagnostician TRUSTWORTHY on each (or the run
is excluded with a reason); `real_sup` reaches EPE on `indoor_flying1` dt = 1 within 2× of the
EV-FlowNet-class numbers in the literature (≈ 0.6–1.0 px) — if not, stop and fix before Phase 6.

---

### Phase 6 — Sim-to-real headline (1 day) ⭐

1. `scripts/run_eval.py` → `results/eval/<arm>/<seed>/<dataset>_<seq>_dt<k>.json` with per-window
   metric arrays (`.npz`, gitignored) and summary JSON (committed).
2. `scripts/make_tables.py` → `results/TABLES.md`:
   - **Table A (T1/T2)**: arm × test sequence × dt, EPE / %Out mean ± 95 % CI, 3-seed mean ± range.
   - **Table B**: gap and closure with paired-bootstrap CIs; self-supervised arms beside supervised.
   - **Table C**: representation ablation.
   - **Table D (T1, generalisation)**: all arms on `dsec_test` (zero-shot from 346×260 training;
     state the resolution change), R1 tagged `reference_seen_data`.
3. Figures: per-sequence EPE bars with CIs; flow colour-wheel triptychs (events / GT / prediction)
   for 6 fixed MVSEC windows (MVSEC imagery is CC BY-SA 4.0 — attribute, share-alike applies to the
   figure).
4. `sim2real-study-runner` builds tables; `eval-reproducibility-auditor` traces each number.

**Exit gate:** Tables A–D complete or each empty cell explained; every row tagged with tier and arm.

---

### Phase 7 — Maritime track (T3) and sea-clutter study (1.5 days, GPU)

1. v2e(SMD on-board) and v2e(LaRS sequences: 9 context frames + keyframe) with `mvsec_cal` and two
   CT variants (×0.67, ×1.5). LaRS context frames come from mixed sources; if the native frame rate
   is not documented per sequence, assume 30 fps, state it, and report LaRS flow in *px per frame
   interval* (rate-free) rather than px/s.
2. `teacher/raft.py`: torchvision RAFT-large on the source frames resized to the event resolution,
   forward and backward, consistency mask (§1.6), stored per window.
3. Train `maritime_selfsup` and `maritime_teacher` (one seed each).
4. Evaluate every model (M0a, M0b, all MVSEC arms, maritime arms, R1 resized) on `smd_test` and
   `lars_eval`: T3a teacher-EPE (with mask coverage) and T3b FWL/RSAT, **sliced by class** on LaRS.
   Also re-evaluate the maritime arms on `mvsec_test` (does maritime training hurt T1?).
5. **Sea-clutter study** (`maritime/clutter.py`) on `lars_eval` keyframe windows: for each class,
   event share vs pixel share, events per pixel per frame interval, and how both move with CT;
   correlation with whichever LaRS scene attributes describe water/glare conditions (inspect
   `image_annotations.json` first; use only attributes that exist). Output curves and a table.
6. Qualitative: side-by-side GIF (frames / events / predicted flow / IWE) for 2 SMD clips — only if
   `docs/LICENSES.md` permits SMD-derived imagery; otherwise LaRS if its terms permit; otherwise
   MVSEC only.

**Exit gate:** T3 tables in their own section of `results/TABLES.md` with tier tags; clutter
results JSON + figure; `sim2real-study-runner` confirms no T3 row sits in a T1/T2 section.

---

### Phase 8 — ONNX export, parity, latency (1 day, GPU)

1. Export M1 (best arm) to ONNX opset 17, fp32 and fp16; the event-representation build stays
   outside the graph (scatter-add in ONNX is fragile) and is benchmarked separately in torch-CUDA.
2. Parity on 200 fixed `mvsec_test` windows: max |Δflow| ≤ 1e-3 px fp32 (TF32 off), ≤ 5e-2 px fp16;
   EPE change ≤ 0.01 px.
3. Latency (§1.6) for representation build at 10⁴/10⁵/10⁶ events, network at 256×256, 346×260
   (padded to /16), 640×480; PyTorch eager, ORT CUDA EP, ORT TensorRT EP fp16. State that the A4000
   stands in for an embedded target.
4. `onnx-parity-auditor`.

**Exit gate:** `results/onnx_parity.json` and `results/latency.json` committed; auditor PASS.

---

### Phase 9 — Validation sweep (1 day)

Walk `VALIDATION_CHECKLIST.md` line by line; verdict + evidence (met / met-with-caveat / unmet) into
`docs/LIMITATIONS.md`. Run `eval-reproducibility-auditor` and `repo-publication-reviewer` (dry run).
Re-run `make reproduce-tables` from committed JSON and diff against `results/TABLES.md`.

**Exit gate:** every line has a verdict; no critical finding open.

### Phase 10 — Publication (0.5–1 day)

README: headline matched-content gap figure + Table B above the fold, evidence-tier legend, a "what
synthetic events can and cannot show" box (§0(1)), maritime section clearly labelled T3, latency
table, limitations, attribution and licences. Release assets: M1 checkpoints + ONNX +
`SHA256SUMS.txt`; **no event data, no frames** in the repo or release. `repo-publication-reviewer`:
SHIP. Tag `phase-10-complete`.

---

## 3. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| v2e too slow for the planned corpus | high | high | Phase 2 throughput budget; cut minutes, never quality settings silently; overnight `/gpu-job` |
| v2e/APS timing misalignment (APS exposure midpoint vs timestamp) | medium | high | Phase 3 cross-correlation check; document the timestamp convention used |
| Calibration over-fits one sequence's statistics | medium | medium | randomisation arm; report fidelity per test sequence |
| `real_sup` fails to reach literature-level EPE | medium | high | Phase 5 gate; compare against R1 and published EV-FlowNet numbers before any sim-to-real claim |
| Event collapse in contrast-maximisation training | high | medium | regulariser + diagnostician; report excluded runs |
| Teacher (RAFT) wrong on water | certain | medium | it is a teacher, not truth; consistency mask; class slices; tier T3a |
| GPL code leaks into the MIT repo | low | high | separate env; `tests/test_no_gpl_imports.py`; publication reviewer |
| Share-alike / unknown dataset terms block README imagery | medium | low | decide in Phase 0; MVSEC figures with attribution as fallback |
| Over-claiming HDR/latency benefits | medium | high | §0(1); `claim-trace` skill word list; reviewer |
| Storage (MVSEC outdoor_day2 + DSEC + LaRS seq 13.6 GB + v2e outputs) | medium | medium | download only listed sequences; v2e outputs compressed (blosc via hdf5plugin) |

## 4. Calendars

- **Full (≈ 11.5 days):** D1 P0 + P1(start) · D2 P1 · D3 P2 · D4 P3 (v2e overnight) · D5 P4 · D6–7 P5 ·
  D8 P6 · D9–10 P7 · D10–11 P8 · D11 P9 · D12 P10.
- **Fast (≈ 6 days):** D1 P0 + P1(start) · D2 P1 · D3 P2-lite + P3-lite (overnight) · D4 P5-lite ·
  D5 P6 + P7-lite · D6 P10 trimmed.

## 5. How this feeds the ship-deck work

- The canonical event store, slicer and representations are sensor-agnostic: a real event camera on
  the Stewart platform or a future hardware buy drops in through one converter.
- The matched-content protocol is exactly how to qualify v2e for the deck problem later: record APS
  + DVS of a deck mock-up once, and the same scripts say how far v2e-trained models can be trusted.
- The sea-clutter study quantifies the event budget a deck-landing pipeline must spend on water —
  an input to sensor choice, ROI masking and bandwidth planning.
- The flow network + latency harness is the front end for divergence/time-to-contact estimation
  in a landing controller (the optic-flow landing line of work: Hérissé et al. 2012 on moving
  platforms; Pijnacker Hordijk et al. 2018 with event-based flow).
