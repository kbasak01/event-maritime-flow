training-run-diagnostician, 2026-10-11, run runs/train/real_sup_voxel5_s7 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/real_sup_voxel5_s7/

Findings:

1. Manifest and git state (INFO, keep). git_sha is 7904922 and git_dirty is false. The worktree ../evflow-wt-phase5b is at HEAD 7904922; its only untracked files are under results/. Seed is 7. train_real_sup.yaml SHA 0c93d7… is the same as in s1337 and s2026. The split SHAs (bdf68aae…, 9ea94aea…) match splits/SHA256SUMS.txt. The GT-cache SHAs, the store SHA (ee3fc241…) and the window counts (10452/10440 train, 2614/2611 tune) are identical across all three seeds.

2. Code drift (LOW, keep). evflow.yaml SHA differs from s1337 and s2026 (8fdfa2… vs 3cdc80…). The only change is `losses.cm.objective` (iwe_variance to average_timestamp), which the supervised path never uses. In src/, drift from f9e996e to 7904922 touches three files:
   - cm.py: CM loss only.
   - loop.py: the change is in the CM branch of batch_loss, plus the manifest's GPU field.
   - data.py: the D-048 end clamp only fires when a store does not cover the range. The real store does, and the manifest has no `end_clamped_us`.
   None of these touch this arm.

3. Curves and LR (INFO, keep). 40 epochs (0–39), no duplicate epochs, no NaN or Inf in any column. train_loss falls steadily from 2.141 to 0.507. LR follows a cosine schedule from 3e-4 (2.995e-4 logged after epoch 0, 0 at epoch 39). grad_norm_max drops from 13.3 to about 2–3. Tune dt1 EPE falls from 0.418 to a plateau of about 0.22 from epoch 20, with no late rise.
   - Train/tune gap: train EPE at the selected epoch is 0.277 (mixed dt1/dt4); tune is 0.218 for dt1 and 0.688 for dt4. The gap is the same across seeds, so it is not seed-specific overfitting.

4. Selection (INFO, keep). argmin real_dt1_epe from metrics.csv is epoch 34 (0.218023). This matches selection.json, the unit log and job.log. best.pt sha256 is ff124e39…1fff, matching selection.json. Selection uses mvsec_tune only. The tune clip [3651, 6275] does not overlap the train clips (first starts at 7588). The only store is outdoor_day2, and mvsec_test is not referenced anywhere in the run's config.

5. Three-seed consistency (INFO, keep). Best dt1 EPE across seeds:

   | Seed | Best dt1 EPE | Epoch | dt4 EPE | dt1 outlier % |
   |---|---|---|---|---|
   | 7 | 0.2180 | 34 | 0.688 | 0.056 |
   | 1337 | 0.2160 | 36 | 0.693 | 0.051 |
   | 2026 | 0.2183 | 36 | 0.695 | 0.050 |

   The spread across seeds is about 1%.

Action: keep. Event-collapse checks do not apply to this supervised arm.
