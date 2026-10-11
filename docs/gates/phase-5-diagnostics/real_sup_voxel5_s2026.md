training-run-diagnostician, 2026-10-10, run runs/train/real_sup_voxel5_s2026 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/real_sup_voxel5_s2026/

**Findings**

1. **Manifest and git state: PASS (info).** Everything is recorded: seed 2026, the 2 config SHAs, git_sha f9e996e, git_dirty false, 4 GT-cache SHAs, the store manifest SHA, the corpus plan SHA, and the train/tune split manifest SHAs. The config SHAs match `git show f9e996e:<file> | sha256sum`. The split SHAs match `splits/SHA256SUMS.txt` and the `manifest_sha256` fields. The worktree HEAD is f9e996e. The only difference from a2b1fa0 under `src/` is the one-line resume change in `loop.py` (`map_location="cpu"`). That path did not run, because this run never resumed. The other changes are the job script (CONCURRENCY=1, SKIP_CM, append-mode logs, `|| true`), the job manifest and DECISIONS. Configs, data provenance, clips and window counts are identical to the s1337 reference. Minor: the worktree has an untracked `results/eval/`, created at 09:45:09 by the stage-1 gate eval, 17 s before this manifest. `git_dirty` evidently ignores untracked files, and that folder holds eval output, not code. Action: keep.

2. **Curves: PASS.** No NaN or Inf and no empty cells in any of the 40 rows. Every epoch has 653 steps. The logged LR matches 0.5·3e-4·(1+cos(π·(e+1)·653/26120)) at every epoch to 4 significant figures, ending at 0. Train loss falls steadily, 2.171 → 0.502. Max grad norm falls 14.0 → about 2.5. Tune flow magnitude settles at about 0.84–0.85. Overfitting is mild: train EPE keeps falling (0.30 → 0.27) while real dt1 tune EPE stays flat at 0.218–0.224 from epoch 24 on. Real dt4 tune EPE is flat at about 0.69–0.70. Action: keep.

3. **Selection: PASS.** argmin of real_dt1_epe from metrics.csv is epoch 36 (0.21830). This matches selection.json (epoch 36, value 0.2182964369014619) and the log line. The rule is min real tune dt1 EPE on mvsec_tune only. Action: keep.

4. **Checkpoint: PASS.** `sha256sum best.pt` = 36e79d368593d1dd7b54e795c0aa843c3c4da41a4e17d392376331a3d81f13a3, which equals selection.json and DONE. best.pt was written at 10:50, just before the epoch-36 log line at 10:51:00.

5. **No test split in loaders: PASS.** `src/evflow/train/data.py` loads only `mvsec_train` and `mvsec_tune`, with `guard_split` checks and `LeakageError` on a GT-cache split mismatch. The manifest shows only train/tune clips (tune clip 3651–6275). Action: keep.

6. **Epoch-11 stall: PASS (low, info).** In the unit log, epoch 11 step 600 was at 10:05:24 and the epoch ended at 10:07:15, about 111 s against the usual ~34 s. So the stall hit mostly the tune-eval/checkpoint phase: epoch_s 179.3 while train_s was only 78.0 (vs about 71 normally). The epoch-11 metrics sit on the trend: loss 0.764, tune EPE 0.2402, grad-norm max 3.9, LR exact. The log shows no restart, error or traceback. Action: keep.

7. **Seed consistency: PASS.** s2026 reaches 0.2183 against 0.2160 for s1337, a gap of 0.0023 px (about 1%). Both select epoch 36, and the per-epoch dt1 curves stay within about 0.01–0.03 of each other from epoch 3 on. Both are far below the zero-flow EPE of 0.892 and fit the expected range for EV-FlowNet-class models (T1, tune). Action: keep.

8. **Collapse checks: not applicable.** This is a supervised run, and its flow magnitude is stable.

**Recommended action:** keep. The run is fit for Phase 5 multi-seed aggregation.
