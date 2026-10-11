training-run-diagnostician, 2026-10-11, run runs/train/real_sup_count_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/real_sup_count_s1337/

**Manifest (OK, keep).** representation=count. n_params is 13,339,464, the same as timesurface (2 channels); voxel5 has 13,341,192 (5 channels). git_sha is 7904922 with git_dirty=false. Worktree HEAD is 7904922, and its only untracked files are under results/. Config SHAs match the worktree's evflow.yaml (8fdfa240…) and train_real_sup.yaml (0c93d720…). Seed is 1337. The train and tune split SHAs (bdf68aae…, 9ea94aea…) match the reference runs. Only the outdoor_day2 store is used: train clips plus tune clip [3651,6275]. No test split appears in the run or the log.

**Selection (OK).** The metrics.csv argmin of real_dt1_epe is epoch 19 (0.22505), which matches selection.json. sha256sum of best.pt is c323fb41…, which matches selection.json and DONE.

**Curves (OK).** No NaN/Inf. Train EPE falls steadily from 1.237 to 0.278. LR follows per-step cosine from 3e-4 to 0 as configured.

**Early epoch 19 (Low, keep).** This is a plateau, not overfitting. After epoch 19, tune dt1 stays between 0.2252 and 0.2409 and ends at 0.2268, only 0.0017 above the best. The epoch-19 value is a small dip within that noise. Tune dt4 drifts up a little (0.740 at epoch 19, 0.769 at the end) while train keeps falling, so there is mild dt4 overfitting. This does not affect the selection rule.

**Grad norms (Low, keep).** Unclipped grad-norm max is 42.5 at epoch 1, against 10.7–26.3 for the reference runs. This fits the unnormalised count inputs. Clipping at 1.0 contained it: max drops below 9 from epoch 9, mean falls from 5.8 to 1.4, and the loss never spiked.

**Comparison (Info).** Selected tune dt1 and dt4 EPE, by run:
- voxel5: 0.216–0.218 dt1 / 0.688–0.695 dt4 (3 seeds)
- voxel15: 0.218 / 0.691
- count: 0.225 / 0.740
- timesurface: 0.233 / 0.773

So count sits between voxel and timesurface, which is plausible for an input with no timing information. Prediction diagnostics are steady and look like the reference runs: tune flow magnitude 0.75–0.97, non-empty IWE fraction about 0.975.

**Action:** keep. In the ablation, report that the count input is unnormalised and that tune dt1 plateaued after epoch 19.
