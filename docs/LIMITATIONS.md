# LIMITATIONS

Written in Phase 9: one row per VALIDATION_CHECKLIST.md line.

| ID | Verdict (met / met-with-caveat / unmet) | Evidence | Note |
|---|---|---|---|

## Known limitations recorded before Phase 9

- **v2e timestamp pyramids (Phase 2, D-025).** v2e runs with a 0.75 ms timestamp cap, chosen in
  Phase 2 as the coarsest that honoured the 60 Hz photoreceptor lowpass check and kept after D-035
  (the lowpass is now off, so no lowpass check runs). Several events per pixel per interpolated
  frame are still spread in v2e's fixed pattern; temporal statistics of v2e events (inter-event
  intervals) carry this artefact and are discussed as such.
- **Symmetric contrast thresholds (Phase 2, D-027).** The plan's calibration grid ties
  pos_thres = neg_thres; the real DAVIS events of the tune slice are OFF-dominated, so the ON-fraction
  statistic cannot be matched by any grid cell.
- **Calibration slice (Phase 2, D-028).** The 32-cell calibration uses 3 × 4 s of `mvsec_tune`, not
  the full 60 s, for compute reasons.
- **Reduced corpus (Phase 2, D-026).** MVSEC arms train on 240 s of `outdoor_day2` (all arms the
  same windows); SMD (45 s train, 30 s test), LaRS (110 train clips) and the clutter study (40 val
  clips) are subsets.
- **LaRS frame rate assumed (Phase 2, D-024).** v2e time constants for LaRS use an assumed 30 fps.
- **Timing-constrained v2e profile: correct event timing at a cost in event statistics (Phase 3,
  T2, D-035).** With the Phase 2 argmin (cutoff_hz 60) v2e events lagged the real events by half an
  APS frame on `mvsec_tune` (count-image NCC-vs-lag peak at +0.5 frame for cell c30,
  `results/v2e_calibration_timing.json`). The frozen profile is now the best lag-free cell, t10
  (thres 0.6, sigma 0.04, cutoff_hz 0): lag peak 0, tune objective (mean normalised W1) 0.701
  against 0.369 for c30 (same file). Turning the photoreceptor lowpass off is what removes the
  lag, so v2e here models no photoreceptor bandwidth; the distribution mismatch it brings is
  accepted and measured, not hidden.
- **The tune-calibrated v2e profile does not reproduce the test recordings' event statistics
  (Phase 3, T2, D-033).** Mean normalised W1 over the six §1.6 statistics, real vs v2e
  (`mvsec_cal`): 0.49 on `mvsec_tune` and 0.48 on the four train clips, but 1.47 (outdoor_day1) to
  2.49 (indoor_flying3) on the test sequences (`results/fidelity/summary.json`). On the indoor
  sequences v2e emits 2.72–2.89× the real event rate and its windows are half ON (0.50) where the
  real DAVIS windows are 0.71–0.74 ON; on outdoor_day1 v2e emits 0.82× the real rate, and the ON
  fraction (real 0.39, v2e 0.51) and the hot-pixel statistic (normalised W1 4.26; real sensor hot
  pixels, which v2e does not model) differ (`results/fidelity/test_*.json`); with the lowpass off
  the hot-pixel statistic also differs on tune and the train clips (normalised W1 1.50 and 1.33,
  `results/fidelity/summary.json`). These are measured
  statistics only; they were not used for any choice. Consequence for Phase 6: the T2 gap on test
  sequences includes this threshold/noise mismatch, which the CT-randomised arm is meant to
  address. Inter-event intervals of v2e carry the timestamp-pyramid artefact (above).
- **What the alignment check can show (Phase 3, T2, D-034, D-035).** Every `mvsec_cal` group
  passes the NCC threshold re-derived on tune (0.491063); indoor_flying2 by the smallest margin
  (median 0.498), and one of its two units, `test/test_indoor_flying2_p0`, is below it on its own
  (median 0.457; the gate is per group, D-034). The 5-frame-shift and x-mirror controls fall below it in every group
  (`results/v2e_alignment.json`). The 1-frame-shift control falls below it on tune and the three
  indoor sequences but not on outdoor_day1 or the train clips, so single-frame timing errors are
  not reliably caught by the threshold; the per-group NCC-vs-lag profile (`lag`) measures timing
  directly: its peak lies at +0.125 frame (one step of its 1/8-frame grid) on tune, train and the
  indoor sequences, and at +0.375 frame on outdoor_day1 (test sequence, reported only: median NCC
  0.675 at lag 0, 0.686 at the peak).
- **Maritime results are T3 only (Phase 7, D-054–D-058).** No ground-truth flow exists for the
  SMD or LaRS video. T3a numbers measure agreement with a frame-based RAFT teacher run on the
  same frames (the teacher is wrong where brightness constancy fails, e.g. on water and glint;
  its forward–backward mask removes only the inconsistent part), T3b numbers measure
  motion-compensation sharpness of the model's own events; neither is accuracy. The events are
  v2e simulations of 8-bit video: they inherit its dynamic range, motion blur, compression and
  30 fps sampling, so nothing in the maritime tables says anything about sun-glare or
  dynamic-range behaviour of a real event sensor (plan §0(1)).
- **Maritime geometry and timing (Phase 7, D-024, D-054).** Every maritime frame is resized by
  v2e to 640×360 without keeping the aspect ratio, so LaRS sources that are not 16:9 are
  stretched (per-unit `native_size` and `scale_xy` in `results/v2e_maritime_corpus.json`); teacher flow, class masks and predictions share
  that grid, and flow is reported in px per frame interval of that grid. LaRS frame rate is an
  assumption (30 fps), so v2e's time constants for LaRS and any per-second rate are assumed too.
- **Small maritime corpus (Phase 7, D-026, D-054).** SMD: every train video over [0, 5.6] s and
  every test video over [0, min(10 s, last frame)] (most videos are too short for D-026's 15 s
  segments); LaRS: 110 train clips and 40 evaluation clips (segment lengths per unit in
  `results/v2e_maritime_corpus.json` `units[].video_s`); the class-sliced and clutter results
  rest on the keyframe windows only (one per clip). Attribute levels with fewer than 3 clips are not
  reported, so rough seas and heavy reflections are mostly unrepresented in the clutter study.
- **Maritime arms selected on driving content (Phase 7, D-056).** CLAUDE.md rule 4 allows
  selection on `mvsec_tune` only, so `maritime_selfsup` and `maritime_teacher` pick their epoch
  on the v2e(mvsec_tune) stream, out of their training domain; the maritime arms are one seed each.
