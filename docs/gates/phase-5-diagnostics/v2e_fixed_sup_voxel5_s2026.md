training-run-diagnostician, 2026-10-10, run runs/train/v2e_fixed_sup_voxel5_s2026 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_fixed_sup_voxel5_s2026/ (I only read files. No torch import, no CUDA process.)

Findings:

1. Manifest, git state and checkpoint: OK. Action: keep.
   - git_sha is 7904922 and git_dirty is false. The worktree was evflow-wt-phase5b.
   - Seed 2026. Config SHAs are present.
   - The split manifest SHAs, corpus_plan SHA, store manifest SHAs (train_cal clips 0-3, v2e tune, real outdoor_day2) and GT cache SHAs are identical to s1337's. The train and tune window counts are also identical.
   - Comparing the manifest with s1337's, the only differences are seed, git_sha, argv path, timestamps, and the evflow.yaml SHA and cm.objective (iwe_variance changed to average_timestamp). The CM objective is not used by the supervised loss.
   - sha256sum of best.pt is d8b4612d…391b008c, which matches selection.json and DONE. best.pt mtime is 23:06:54, the same second as the epoch 29 log line.

2. Code drift from a2b1fa0 to 7904922: no effect on this arm. Action: keep.
   - loop.py changes only the CM branch (polarity passed to CM), the resume path (load to CPU) and the manifest "gpu" field.
   - data.py adds an end clamp. It did not trigger: there is no end_clamped_us key in the manifest provenance and the window counts are unchanged.
   - No resume happened: one session from 22:00 to 23:29 (5388 s for 40 epochs), with no resume, clamp, error or traceback in the log.

3. Curves and LR: OK. Action: keep.
   - 40 rows and 653 steps per epoch. No NaN or Inf in any column.
   - Train loss falls steadily, 2.255 to 0.505.
   - LR follows cosine from 3e-4 to 0 (max deviation 2e-16).
   - Grad norm falls from 3.7 to 1.0.
   - v2e dt1 tune EPE levels off at about 0.24 from epoch 24, while train keeps improving. That is mild flattening, not overfitting.

4. Selection: OK. Action: keep.
   - argmin of v2e_dt1_epe in metrics.csv is epoch 29 (0.23443), which matches selection.json exactly.
   - Only mvsec_tune is used. The manifest has no reference to test splits (outdoor_day1, indoor_flying).

5. Real vs v2e tune gap (informational, T2):
   - At epoch 29, real dt1 is 0.293 vs v2e 0.234 (+25%). Real dt4 is 0.851 vs v2e 0.701.
   - real_sup selects at 0.216 (s1337) and 0.218 (s2026) on real tune dt1, so v2e_fixed is about 36% worse on real events.
   - These values look plausible for EV-FlowNet-class models on the center_256_no_hood region.

6. Seed agreement with s1337: good.
   - Selected value 0.2344 vs 0.2339 (epoch 29 vs 26).
   - Real dt1 at the selected epoch 0.293 vs 0.300. Final flow_mag 4.09 vs 4.16.
   - Only epoch 0 differs (v2e dt1 0.56 vs 0.43), which is not material.
   - Reference: v2e_rand_sup s1337 selects at 0.232.

Low: the job manifest is the only evidence that the run was alone on the GPU. Epoch times (about 124–142 s) are steady and show nothing wrong. No action needed.
