# LIMITATIONS

Written in Phase 9: one row per VALIDATION_CHECKLIST.md line.

| ID | Verdict (met / met-with-caveat / unmet) | Evidence | Note |
|---|---|---|---|

## Known limitations recorded before Phase 9

- **v2e timestamp pyramids (Phase 2, D-025).** v2e runs with a 0.75 ms timestamp cap, the coarsest
  that honours its 60 Hz photoreceptor lowpass check. Several events per pixel per interpolated
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
- **The tune-calibrated v2e profile does not reproduce the test recordings' event statistics
  (Phase 3, T2, D-033).** Mean normalised W1 over the six §1.6 statistics, real vs v2e
  (`mvsec_cal`): 0.26 on `mvsec_tune` and 0.30 on the four train clips, but 1.47 (outdoor_day1) to
  1.94 (indoor_flying1) on the test sequences (`results/fidelity/summary.json`). On the indoor
  sequences v2e emits 2.98–3.42× the real event rate and its windows are about half ON (0.49–0.50)
  where the real DAVIS windows are 0.71–0.74 ON; on outdoor_day1 the rates agree (ratio 0.98) but
  the ON fraction (real 0.39, v2e 0.51) and the hot-pixel statistic (normalised W1 4.11, real
  sensor hot pixels that v2e does not model) differ (`results/fidelity/test_*.json`). These are
  measured statistics only; they were not used for any choice. Consequence for Phase 6: the
  T2 gap on test sequences includes this threshold/noise mismatch, which the CT-randomised arm is
  meant to address.
- **What the alignment check can show (Phase 3, T2, D-034).** Every `mvsec_cal` group passes the
  frozen NCC threshold, and the 5-frame-shift and x-mirror controls fall below it in every group
  (`results/v2e_alignment.json`). The 1-frame-shift control does not: its median stays within 0.05
  of the aligned median in every group, so the check cannot detect a timing error of a single APS
  frame on these scenes. Windows in which the vehicle stands still (a stretch late in the tune
  slice, found by inspection at the Phase 3 gate) contain mostly noise events and have near-zero
  NCC; they lower the tune p10, not the median.
