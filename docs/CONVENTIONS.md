
# Conventions (canonical copy: docs/CONVENTIONS.md — keep identical)

## Event store (HDF5)
- `events/x`, `events/y`: uint16, origin top-left, x → right, y → down.
- `events/t`: int64 microseconds, non-decreasing, relative to `t_offset` (int64 µs).
- `events/p`: int8, 0 = OFF (brightness decrease), 1 = ON. Converters map ±1 or bool sources to this.
- `ms_to_idx[ms]`: first index with `t >= ms*1000`; `t[ms_to_idx[ms]-1] < ms*1000` (DSEC definition).
- Window slicing is half-open: `t_start <= t < t_end`.
- Attrs: `width`, `height`, `source`, `sensor`, `distortion` ∈ {raw, rectified, undistorted},
  `polarity_convention`, and for synthetic data `simulator_manifest_sha256`.

## Flow
- Forward flow from window start to window end, in **pixels over the window** (not px/s) unless a
  field is explicitly named `*_px_per_s`.
- Array layout `(2, H, W)` with channel 0 = u (x, right), 1 = v (y, down).
- Horizontal image flip ⇒ u ← −u after flipping; vertical flip ⇒ v ← −v.
- MVSEC dt = k means the window spans k APS frame intervals. Never average across dt values.
- DSEC flow PNG: `u = (R − 2^15)/128`, `v = (G − 2^15)/128`, valid = `B == 1`.

## Masks
- Every metric takes explicit masks: `valid_gt`, optional `events_in_window` (MVSEC sparse),
  optional `crop`, optional `class_mask` (maritime). No hidden defaults inside metric functions.

## Metrics names
- `epe` (T1/T2 only), `outlier_pct` (EPE > 3 px AND > 5 % of |GT|), `npe_{1,2,3}`, `ae_deg`,
  `fwl` (> 1 better; zero flow = 1), `rsat`, `teacher_epe` (T3a only, with `teacher_mask_coverage`).

## Results JSON (every summary file)
Required keys: `tier` ∈ {T1, T2, T3a, T3b}, `arm`, `model`, `seed`, `checkpoint_sha256`,
`dataset`, `sequence`, `dt` (or `window_ms`), `representation`, `metrics` (dict),
`ci95` (dict, where applicable), `n_windows`, `git_sha`, `created_by` (script path), `created_at`.
Tags `reference_external` / `reference_seen_data` for R1 rows.

## Evidence tiers
- T1 real events + real GT · T2 v2e from same-sensor frames + real GT · T3a maritime vs RAFT teacher
  · T3b maritime self-consistency only. T3 never shares a table section with T1/T2.

## Words not to use about v2e results
"HDR", "glare-robust", "low-latency advantage", "real-world accuracy" (for T3), "ground truth"
(for teacher flow), "guaranteed", "first" (unless §0 positioning supports it).
