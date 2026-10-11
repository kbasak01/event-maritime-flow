training-run-diagnostician, 2026-10-11, run runs/train/v2e_rand_sup_voxel5_s7 (verbatim)

VERDICT: TRUSTWORTHY

Run dir: runs/train/v2e_rand_sup_voxel5_s7/. I made no edits, started no CUDA process and did not import torch.

Findings:

1. Manifest (INFO, keep). The run used commit 7904922 with git_dirty=false, launched from evflow-wt-phase5b with seed 7. Both config SHAs match the files in the main repo and the worktree. The train/tune split SHAs (bdf68aae / 9ea94aea) match splits/SHA256SUMS.txt; sha256sum of the files themselves differs because each file embeds its own hash, as in the sibling runs. Each train_clip0..3 has 4 draws (rand0..3, seeds 2000-2303) with their CT overrides. data.end_clamped_us = 115 us for rand1 and rand2 train_clip2, as D-048(a) says. Window counts are 10452/10440 (train) and 2614/2611 (tune). Apart from the seed, the manifest's config and data blocks are identical to s1337 and s2026.

2. Curves and LR (INFO, keep). 40 epochs in one session (log has 40 epoch lines and no NaN, error or resume). No NaN or Inf in metrics.csv. Train loss falls from 2.24 to 0.53, and grad norm from 3.65 to 1.00. LR matches a per-step cosine from 3e-4: 2.995e-4 after epoch 0, 1.5e-4 at epoch 19, 0 at epoch 39. Tune EPE levels off at about 0.231 from epoch 27 with no upturn, so no overfitting. Train EPE (0.289) is above tune because training mixes in dt4 windows and uses augmentation.

3. Selection (INFO, keep). argmin of v2e_dt1_epe is epoch 33 (0.230061), which matches selection.json and the log. Epoch 31 is only 6e-5 worse, so the pick is rule-correct but noise-level. best.pt SHA-256 f8de9fb7... matches selection.json and DONE. Only mvsec_tune is used; the manifest has no test split or test sequence names.

4. Real vs v2e tune gap (INFO, report). At the selected epoch, real dt1 EPE is 0.296 against 0.230 on v2e (+29%), and dt4 is 0.835 against 0.697. real_sup s7 reaches 0.218 on real tune, so the sim-to-real gap is visible.

5. Three seeds agree (INFO, keep):
   - v2e dt1: 0.2301 / 0.2323 / 0.2344 (s7 / s1337 / s2026)
   - real dt1: 0.2962 / 0.3006 / 0.2966
   - For comparison, v2e_fixed_sup s7 got 0.2369 on v2e and 0.3015 on real.

Action: keep. The run joins the other two v2e_rand_sup seeds.
