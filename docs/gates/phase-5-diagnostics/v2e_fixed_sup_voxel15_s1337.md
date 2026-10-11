training-run-diagnostician, 2026-10-11, run runs/train/v2e_fixed_sup_voxel15_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_fixed_sup_voxel15_s1337/

**Checks that pass (keep)**
- **Manifest:** representation is voxel15, n_params is 13,346,952 (identical to real_sup_voxel15_s1337; voxel5 is 13,341,192), arm v2e_fixed_sup, seed 1337. git_sha is 7904922 with git_dirty false; the worktree HEAD matches, and its only untracked files are results/. The split manifest SHAs (train bdf68aae, tune 9ea94aea) and the train_yaml SHA match the voxel5 cell. There is no end clamp.
- **Code since the voxel5 references (a2b1fa0):** only the CM objective, the CM polarity plumbing, the CPU RNG resume fix and the rand-store end clamp changed. None of these touch the supervised path on the fixed stores.
- **Data:** only train_cal clips, the v2e tune store and real outdoor_day2 (tune clip 3651–6275) appear. The manifest and log contain no test sequence.
- **Curves:** 40 epochs, no NaN or Inf. Training loss falls from 2.28 to 0.52. Per-step cosine LR runs from 3e-4 to 0, logged at the end of each epoch exactly as in the references. Max grad norm falls from 8.2 to about 3. Tune EPE has one brief spike at epoch 3 (0.72) and recovers by epoch 4.
- **Selection:** recomputed argmin of v2e_dt1_epe is epoch 26 (0.24022). This matches selection.json, the log and DONE.
- **Checkpoint:** sha256sum best.pt gives f4546067…885051e, matching selection.json and DONE.
- **Overfitting:** none. Tune EPE stays flat (0.240–0.249) from epoch 19 while training EPE keeps falling. Training EPE (0.30) is a dt1/dt4 mix with augmentation, so it is not directly comparable to tune EPE.

**Findings (low severity, record only)**
1. **Slightly worse than the voxel5 seed spread.** Each pair below is voxel15 vs. the voxel5 range over 3 seeds, at the selected epoch:
   - v2e dt1: 0.240 vs 0.234–0.237
   - real dt1: 0.309 vs 0.293–0.301
   - v2e dt4: 0.726 vs 0.701–0.714
   - real dt4: 0.934 vs 0.851–0.874

   This is one seed, so it reads as "no gain from 15 bins". It should not be presented as a significant drop.
2. **Real-vs-v2e tune gap:** at dt1 it is 0.069 px vs 0.059–0.066 px for voxel5, so the sim-to-real gap is about the same. The real_sup voxel15 cell's own real tune dt1 is 0.218.
