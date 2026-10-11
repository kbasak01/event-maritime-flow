training-run-diagnostician, 2026-10-11, run runs/train/real_sup_timesurface_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/real_sup_timesurface_s1337/

Findings:
1. Manifest: OK (info). Representation is timesurface. n_params is 13,339,464, which equals 13,341,192 − 1,728, so the input has 2 channels. git_sha is 7904922 with git_dirty false, and HEAD of the worktree matches. Seed is 1337. The config SHAs, both split-manifest SHAs, gt_cache, tune_clip and train_windows all match voxel5_s7 (same commit). The resolved train config differs from voxel5 only in `representation`. Action: keep.
2. Curves and LR: OK (info). No NaN or Inf in any of the 40 rows. Train loss falls steadily from 2.19 to 0.54. The cosine LR goes from 3e-4 to 0, as configured. The log has no errors, resumes or "test" entries.
3. Selection: OK (info). The argmin of real_dt1_epe is epoch 26 (0.23315), which matches selection.json. The SHA-256 of best.pt is 5239869d…5334, which matches selection.json and DONE. Only mvsec_tune appears, with no test split anywhere.
4. Worse tune EPE looks like a representation effect, not a training problem (info). Best tune dt1 EPE is 0.233. The voxel references sit at 0.216–0.218 (voxel5 seeds 1337/2026/7 and voxel15), so this run is about 7% worse and well outside the voxel seed spread. There is only one timesurface seed, though.
   - Train EPE is also higher (0.288 vs 0.273 at epoch 39), so it fits training data worse too. That points to less input information, not overfitting.
   - Tune EPE plateaus from about epoch 23 and only drifts slightly after epoch 26 (last epoch 0.237).
   - Grad norms are somewhat higher (mean 1.4 vs 1.0) but have no spikes, so this is not instability.
   - The dt4 gap is larger (0.77 vs 0.69). That fits a last-timestamp surface losing event multiplicity over long windows.
5. tau plumbing (low). The config field `timesurface.tau_fraction_of_window: 0.333` is never read. `make_input` defaults to 1/3 and `batch_inputs` hard-codes `dur / 3.0` (../evflow-wt-phase5b/src/evflow/train/data.py:410). The values are effectively the same (0.1% apart), but the manifest is misleading. Action: keep the run, but either wire the config value through or note in docs/DECISIONS.md that it is hard-coded.

Recommended action: keep. When reporting the ablation, say it is a single seed.
