---
name: v2e-convert
description: "Policy and procedure for converting frames or video to DVS events with v2e in this repo (pinned args, frame rate, seeds, manifests, trial runs, conversion to the canonical event store). Use whenever writing or running anything that calls v2e."
argument-hint: "[source] [clip-id] [profile]"
---

# v2e conversion: $ARGUMENTS

## Rules
- v2e runs only in the `v2e` conda env, only via `src/evflow/sim/v2e_runner.py`, only with a
  profile from `configs/v2e_profiles.yaml`. No ad-hoc flags in code.
- Always pass: `--no_preview --skip_video_output --overwrite --dvs_h5 <out>.h5
  --dvs_emulator_seed <seed>` and an explicit `--input_frame_rate` (from `ffprobe -v error
  -select_streams v:0 -show_entries stream=r_frame_rate` for videos, from the dataset's timestamps
  for MVSEC APS, from the documented/assumed rate for LaRS — log the assumption).
- MVSEC APS: `--dvs346`. Maritime: `--output_width 640 --output_height 360` unless the profile says
  otherwise. Use frame folders (`-i <folder>`) for MVSEC/LaRS; videos for SMD.
- MVSEC APS timestamps are not uniform: export frames + their timestamps, and after conversion
  re-map v2e's uniform clock to the real APS clock (piecewise-linear between frame timestamps).
  Record the mapping error per clip. (v2e's `--synthetic_input` accepts per-frame times but
  disables SuperSloMo interpolation, so it is not the default route; use it only as an ablation.)
- New profile or new source ⇒ trial first with `--stop_time 5` (v2e's own advice), inspect event
  rate and a count image, then scale.

## Procedure
1. Resolve source paths from `configs/paths.local.yaml`; compute input SHA-256.
2. Build argv = `common` + profile `size` + profile `args` + per-run flags; show it.
3. `conda run -n v2e python <v2e.repo>/v2e.py <argv>`; capture stdout/stderr to the run log
   (v2e prints undersampling warnings for the lowpass filter — grep for them and fail if present).
4. Inspect the h5 (`h5ls -v`) the first time; convert to the canonical store with
   `io/v2e_reader.py`; apply the clock re-map if MVSEC.
5. Write the manifest next to the output: v2e commit, argv, seed, input/output SHA-256, frame-rate
   source, wall time, GPU name, warnings.
6. Run the `event-data-auditor` checks (monotonic t, polarity set, bounds) on the output.

## Known v2e artefacts to keep in mind
- Multiple events per pixel per (interpolated) frame are spread in a fixed "pyramid" pattern between
  frames; reduce with a finer timestamp resolution, at a compute cost.
- Output inherits the source's dynamic range, blur and compression artefacts — never claim HDR.
