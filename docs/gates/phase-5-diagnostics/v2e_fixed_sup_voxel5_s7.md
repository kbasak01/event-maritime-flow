training-run-diagnostician, 2026-10-11, run runs/train/v2e_fixed_sup_voxel5_s7 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_fixed_sup_voxel5_s7/. I worked read-only and started no CUDA process.

**Findings**

1. **Manifest and git: OK (info).**
   - Commit 7904922, `git_dirty: false`, seed 7, launched from the evflow-wt-phase5b worktree, whose HEAD is 7904922. Its only untracked paths are the results/ outputs.
   - Both config SHAs match the files on disk in the main repo and the worktree.
   - Split SHAs match `splits/SHA256SUMS.txt` (bdf68a… train, 9ea94a… tune). They are canonical-JSON hashes by design (make_splits.py), so they differ from the raw file sha256.
   - Compared with the s2026 manifest, only the seed and run_id differ. Compared with s1337, the only difference is the CM objective name (D-047), which a supervised run does not use.
   - Action: keep.

2. **Curves: OK (info).**
   - 40 epochs × 653 steps, no NaN/Inf.
   - Train loss falls 2.262 → 0.510; grad-norm max falls 13.5 → 2.8.
   - LR exactly equals a per-step cosine from 3e-4 to 0 (max deviation 0).
   - The unit log shows epochs 0–39 in one session (no resume). Tune flow magnitude stays between 0.85 and 1.26.

3. **Selection: OK (info).**
   - Argmin of `v2e_dt1_epe` is epoch 34 (0.23688), which matches selection.json and the `selected_so_far` column.
   - sha256sum of best.pt is 8a584851…cd0, matching selection.json.
   - No test sequence or test split appears in the manifest; the only real store is outdoor_day2 tune clip [3651, 6275].

4. **Real-vs-v2e tune gap (low, expected).**
   - At epoch 34: v2e dt1 0.237 vs real dt1 0.301 (+27%); dt4 0.714 vs 0.874.
   - Real dt1 is lowest at epoch 20 (0.292), but it is logged only and correctly not used for selection.
   - v2e dt4 drifts up slightly after epoch 20 (0.695 → 0.71), which is mild.
   - Action: keep, and report the gap as sim-to-real.

5. **Three-seed consistency: OK (info).** The epoch-0 tune EPE is higher for s7 (0.635 vs 0.43 and 0.56), but it converges by epoch 8.

| Seed | Selected epoch | v2e dt1 | real dt1 | Final train EPE |
|---|---|---|---|---|
| s7 | 34 | 0.237 | 0.301 | ≈0.27 |
| s1337 | 26 | 0.234 | 0.300 | ≈0.27 |
| s2026 | 29 | 0.234 | 0.293 | ≈0.27 |

6. **GPU exclusivity: OK.** In job.log, real_sup_s7 finished at 02:14:02, this run started at 02:14:10, and the next unit (v2e_rand_sup_s7) started at about 03:44 after this one ended. No other process overlapped.

Overall action: keep.
