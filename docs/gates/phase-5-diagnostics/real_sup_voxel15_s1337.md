training-run-diagnostician, 2026-10-11, run runs/train/real_sup_voxel15_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/real_sup_voxel15_s1337/

Findings:

1. Manifest and git: no issue. Action: keep.
   - The manifest records `representation: voxel15`. argv includes `--representation voxel15`, and the resolved config has `train.representation: voxel15`.
   - n_params is 13,346,952 against 13,341,192 for voxel5. The difference of 5,760 equals 10 extra input channels × 64 outputs × 3×3 kernel in the first encoder conv (no bias), so the input really has 15 channels and nothing else in the model changed.
   - Git is 7904922 and clean. Seed 1337. Split SHAs match the voxel5 reference: tune 9ea94aea…, train bdf68aae….
   - The evflow.yaml SHA differs from voxel5_s1337 (built at a2b1fa0). The only change between those commits is the CM objective line, which a supervised run never uses. data.py also changed (the D-048a rand-store end clamp), but the window counts are identical (10,452 / 10,440 train, 2,614 / 2,611 tune).

2. Curves: no issue. Action: keep.
   - 40/40 epochs and 40 epoch lines in the unit log, with no resume, error, NaN or Inf.
   - Train loss falls from 2.17 to 0.52. Tune dt1 EPE falls from 0.391 to 0.218. Grad-norm max is 10.7.
   - The LR follows per-step cosine from 3e-4 to 0, matching the voxel5 runs exactly.
   - Train EPE (0.274) mixes dt1 and dt4 windows, so it is not directly comparable to tune dt1 EPE. There is no sign of overfitting: tune EPE plateaus around 0.218–0.222 from epoch 26 on.

3. Selection: no issue. Action: keep.
   - argmin of real_dt1_epe in metrics.csv is epoch 36 (0.218199). This matches selection.json, the `selected_so_far` column and the log line.
   - sha256sum of best.pt is 6653f458…6098, matching selection.json and DONE.
   - Selection uses `mvsec_tune` only. The training data loaders contain no test-split reference.

4. Comparison with voxel5 (minor). Action: keep, and report as no detectable difference.
   - The voxel5 3-seed tune dt1 spread is 0.2160 (s1337), 0.2180 (s7) and 0.2183 (s2026). voxel15 at 0.2182 falls inside it, at the upper edge, and is +0.002 px against the same-seed voxel5 run.
   - dt4 EPE at the selected epoch is 0.691, also inside the voxel5 range of 0.688–0.695.
   - This is a single seed (seed 1337 only, per D-043f), so it cannot resolve differences smaller than the seed spread of about 0.002 px.

5. Event-collapse check: not applicable, since this is a supervised arm. Tune flow magnitude stays stable at 0.74–0.84.
