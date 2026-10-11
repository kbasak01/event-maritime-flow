training-run-diagnostician, 2026-10-11, run runs/train/v2e_fixed_sup_timesurface_s1337 (verbatim)

VERDICT: TRUSTWORTHY

I checked this read-only. I did not start a CUDA process or import torch.

1. **Manifest and git: OK.** representation=timesurface, n_params=13,339,464 (matches real_sup_timesurface; voxel5 has 13,341,192). Seed is 1337. git_sha=7904922, git_dirty=false. The worktree HEAD is 7904922 with only untracked results/ files. Both config SHAs match the files on disk. The split, corpus_plan and store manifest SHAs, and tune_clip [3651,6275], are identical to the trusted v2e_fixed_sup_voxel5_s1337. No test split appears: the data is train_cal clips, the v2e tune store and real outdoor_day2 tune. Action: keep.

2. **Curves: OK.** 40 epochs × 653 steps, no NaN/Inf in any column. Train loss falls 2.33→0.54 and train EPE 1.25→0.29. The LR matches cosine from 3e-4 to within 5e-20. Predicted flow magnitude levels off around 4.1, so it is not growing without bound. Max gradient norm falls from 24.7 to about 5. Tune EPE was noisy early (spikes at epochs 2 and 8) and then flat between 0.26 and 0.28. Train EPE at selection is 0.315 against a v2e tune EPE of 0.2625, so there is no overfitting. Severity low. Action: keep.

3. **Selection: OK.** The argmin of v2e_dt1_epe is epoch 26 at 0.2625, which matches selection.json and the unit log. sha256sum of best.pt gives 91a81393…89f0, matching selection.json. Real tune is logged only; its minimum (0.321 at epoch 32) played no part in the choice. Action: keep.

4. **Representation effect, not a training problem.** At the selected epochs (EPE, dt1):

| Run | v2e tune | real tune | real/v2e |
|---|---|---|---|
| timesurface s1337 (this run) | 0.2625 | 0.334 | 1.273 |
| voxel5 s1337 / s2026 / s7 | 0.234–0.237 | 0.293–0.302 | 1.25–1.28 |
| voxel15 s1337 | — | — | 1.284 |

- Timesurface is about 12% worse than voxel5 on both v2e and real tune. That is larger than the voxel5 spread across seeds, but it is the same size on both streams.
- The sim-to-real gap is the same as voxel's: the real/v2e ratio sits inside the voxel5 seed range.
- The real arm's timesurface penalty is about 8% (0.233 vs 0.216). Same direction, slightly smaller.

So the deficit is the input representation, not a training failure or extra sim-to-real loss. Severity info. Action: keep, and report it with the T2 tag.

5. **tau hard-coded at window/3 (D-050).** This is already recorded and does not affect validity. Severity info.
