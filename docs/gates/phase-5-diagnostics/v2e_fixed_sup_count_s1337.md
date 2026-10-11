training-run-diagnostician, 2026-10-11, run runs/train/v2e_fixed_sup_count_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run directory: runs/train/v2e_fixed_sup_count_s1337/. I read its files and the unit log only. No torch import, no CUDA.

**Integrity (keep):**
- Manifest: `representation=count`, `n_params=13339464`, seed 1337, `git_sha` 7904922, `git_dirty=false`. The worktree HEAD is 7904922 and has no tracked changes; only untracked `results/` files are present.
- Both config SHA-256s match the files on disk.
- Data: the 4 train_cal clips, v2e tune store and real `outdoor_day2` store, all with manifest SHAs. "test" appears nowhere in the manifest or the log.
- The SHA-256 of `best.pt` is bd3ca679…6aa0, which matches `selection.json` and `DONE`.

**Curves (keep):**
- 40/40 epochs at 653 steps each, no NaN or Inf in any column.
- The LR follows the cosine from 3e-4 to 0 as configured.
- Training loss falls smoothly from 2.61 to 0.52.
- It ran as a single session: 5051 s is about 40 epochs of ~126 s each.

**Selection (keep):** argmin of `v2e_dt1_epe` is epoch 26 at 0.2471, which matches `selection.json`. Selection used only `mvsec_tune` v2e events. Real-tune metrics are logged but not used for the choice.

**Gradient norms (low, note only):** with the unnormalised count input, pre-clip norms are higher than voxel5's.
- Epoch-0 mean is 4.40 vs 3.54; the run's maximum is 18.3 vs 14.4.
- The final-epoch mean is 1.34 vs 1.02, so most steps are clipped at 1.0.
- This is consistent with the real-arm count cell (max 42.5, its tune EPE 0.225, judged trustworthy). There is no instability in loss or tune curves.

**Overfitting (none):** v2e-tune dt1 EPE flattens near 0.25 from about epoch 19 while training EPE keeps falling to 0.27. Tune EPE never turns back up; the last epoch is 0.250.

**§1.5 hypothesis (report, do not assume):** the data do not support "count images are less sensitive to v2e timestamp artefacts." At the selected epoch:

| Arm | v2e tune | real tune | gap | real ÷ v2e | dt4 gap |
|---|---|---|---|---|---|
| count | 0.247 | 0.330 | 0.083 | 1.33 | 0.211 |
| voxel5, 3 seeds | | | 0.059–0.066 | 1.25–1.28 | 0.151–0.162 |
| voxel15 | | | 0.068 | 1.28 | 0.208 |
| timesurface | | | 0.072 | 1.27 | 0.268 |

- The sim-to-real cost against the real arm is also larger for count (0.330 vs 0.225, +0.105) than for voxel5 (0.300 vs 0.216, +0.084).
- Real-tune outliers are 0.42% vs 0.09% on v2e tune.
- Caveat: count is a single seed, while voxel5's seed spread is about 0.007. Record it as "no evidence for the hypothesis; count slightly worse."

**Action:** keep and use as the count cell.
