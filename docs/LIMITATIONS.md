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
