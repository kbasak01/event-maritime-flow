training-run-diagnostician, 2026-10-10, run runs/train/v2e_rand_sup_voxel5_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_rand_sup_voxel5_s1337/

**Manifest (OK).** git 7904922, git_dirty=false. argv points to the clean worktree evflow-wt-phase5b. Seed 1337. The config SHAs (evflow.yaml 8fdfa240…, train_v2e_rand_sup.yaml 984591b7…) match `git show 7904922:` of each file. The train/tune split SHAs, gt_cache SHAs, train_clips, tune_clip and the tune stores (v2e tune + real outdoor_day2) are identical to v2e_fixed_sup. All 16 rand{0..3}/train_clip{0..3} stores are listed with manifest SHA, v2e seed (2000–2303) and pos/neg/sigma CT overrides. `data.rand_draws` matches them. `end_clamped_us` = 115 µs for rand1 and rand2 train_clip2 only. Window counts are 10452 dt1 / 10440 dt4 train and 2614/2611 tune, the same as v2e_fixed_sup. steps/epoch 653 and n_params 13,341,192 also match.

**Code delta vs the reference (a2b1fa0 → 7904922, OK).** In the supervised path the only change is the D-048a end clamp in `from_store` (the rest is the CM polarity change and the CPU resume load, neither used here). The sampler draws one of the K stores per sample (`torch.randint(len(cd.sources))`, data.py:265).

**Curves (OK).** 40 epochs logged. No NaN or Inf anywhere in metrics.csv. Train loss falls monotonically from 2.236 to about 0.53. The LR matches the per-step cosine from 3e-4 to 0 at every epoch end. grad_norm_max peaks at 18.9 (epoch 1), then stays at 2.5–4. Tune v2e dt1 EPE stays flat at 0.232–0.242 from epoch 22 to 39 and never rises, so there is no overfitting. Train EPE mixes dt1 and dt4 50/50, so it cannot be compared directly with tune dt1. Single session: wall time 5430.8 s ≈ 40 × ~130 s, so the run did not resume.

**Selection (OK).** Recomputed argmin of v2e_dt1_epe = epoch 28 (0.23228). This equals selection.json and the log's last line. `sha256sum best.pt` = 8aa40f3225b55406136c4a2dd8edd1ce7db89e91a9264ed72815eb39a9b75686, which matches selection.json. No test sequence appears in the manifest (0 matches for "test"); the only data used is mvsec_train and mvsec_tune.

**Real-vs-v2e tune gap (OK, same as reference).** At the selected epoch:
| Run | v2e dt1 | real dt1 | gap | dt4 v2e / real |
|---|---|---|---|---|
| rand | 0.2323 | 0.3006 | +0.068 (×1.29) | 0.698 / 0.827 |
| fixed (epoch 26) | 0.2339 | 0.2995 | +0.066 (×1.28) | 0.719 / 0.887 |

Real dt1 EPE is about 3× better than zero flow (0.892). The best real-tune epoch would have been 27 (0.2975); it was not used, as D-043b requires. Predicted flow magnitude on real tune (0.79) is a little below v2e (0.97), the same pattern as fixed. Collapse checks do not apply to this supervised arm.

**Minor findings.**
1. (Info) The unit log runs/phase5-train/train_v2e_rand_sup_voxel5_s1337.log opens with the D-048a traceback: "rand1/train_clip2.h5 … not covered", from the old worktree evflow-wt-phase5. The traceback has no timestamp and no training step comes before it. The real run starts at 15:53 and runs without a break to the selection line at 17:23. Action: keep. If the 09:45 time matters for the record, take it from job.log.
2. (Low) D-048a says the stores end 369 µs short, but the recorded clamp is 115 µs. That is consistent: the clamp is measured against the ms_to_idx covered end, not the last event. Action: keep. One clarifying clause in DECISIONS is optional.
3. (Low) Store `n_events` is null (also in the reference), and which store each sample used is not logged; it can be rebuilt from the seeded sampler. Action: keep.

Overall recommendation: keep. The run is fit to be the v2e_rand_sup voxel5 s1337 cell.
