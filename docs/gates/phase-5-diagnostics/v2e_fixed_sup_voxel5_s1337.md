training-run-diagnostician, 2026-10-10, run runs/train/v2e_fixed_sup_voxel5_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_fixed_sup_voxel5_s1337/

**What checks out**
- **Manifest is complete and matches the files on disk.** git a2b1fa0 (clean). Since then HEAD has changed only the resume path in loop.py, and this run never used it. The SHAs of both config files match the current files. The split manifests (train bdf68aae…, tune 9ea94aea…) match splits/SHA256SUMS.txt. The corpus_plan, all 5 v2e store manifests, the real store manifest and all 4 GT-cache .h5 files hash to the values recorded in the manifest. Seed is 1337 and the arm is train_cal / tune_select v2e / tune_log real.
- **Checkpoint is the selected one.** sha256(best.pt) = 9dd9c585…651f, which equals selection.json and DONE. best.pt stores epoch=26. last.pt has epoch=39 and best={26, 0.23394}.
- **Selection is correct.** The argmin of the v2e_dt1_epe column is epoch 26 (0.23394), which matches selection.json. If the real stream had been used instead, the argmin would be epoch 32 (0.2983 vs 0.2995 at epoch 26), so the choice is not sensitive to which stream selects.
- **No test split can enter.** `guard_split` in data.py only accepts `mvsec_train` and `mvsec_tune` and checks APS ranges against the split manifest. The GT caches check their split/dt attributes. Train clips are cross-checked against the corpus plan. The manifest lists only train_cal clips, the tune v2e store and the real outdoor_day2 store (used for the tune slice, APS 3651–6275).
- **Curves are clean.** 40 rows, no NaN, Inf or blanks. Training loss falls steadily from 2.209 to 0.518. Mean gradient norm falls from 3.5 to about 1.0, with no spikes after epoch 2. Tune metrics are logged every epoch for both streams. The logged LR matches per-step cosine to 0 over 40×653 steps, with a maximum error of 5e-20. There is no warm-up; that is as configured, since `warmup_windows` means dropped windows, not LR warm-up.
- **The incident did not affect this run.** wall_time_s_this_session is 8206.6 s, which covers the whole 06:28–08:45 span, so there was a single session with no restart. Epoch 20 spans the crash at 07:58. Its loss is continuous with epoch 19 (0.6427 both), its tune EPE is 0.2518 (within noise), and only its epoch time changes (236 s to 134 s, consistent with now having the GPU alone).

**Findings**
1. **Medium: the unit log has NOT been overwritten yet, but it will be.** runs/phase5-train/train_v2e_fixed_sup_voxel5_s1337.log is intact: 10.5 KB, all 40 epochs, no errors, and it ends with the selection line. The relaunched job (job.log) is still in stage 1 on real_sup. When it reaches this unit, `> "$LOGS/train_$name.log"` will replace the log with the "DONE present, skipping" line. Action: copy the log aside now (e.g. `*.attempt1.log`) and note it in D-045.
2. **Low: dt4 overfits mildly; dt1 has plateaued.**
   - v2e dt4 EPE bottoms at 0.684 (epoch 12) and is 0.709 at the selected epoch (+4%).
   - real dt4 EPE bottoms at 0.807 (epoch 14) and is 0.871 at the selected epoch (+8%).
   - v2e dt1 improves only from 0.247 (epoch 12) to 0.234, while train EPE keeps falling (0.415 to 0.27).
   - train_epe mixes dt1 and dt4 over random crops, so it cannot be compared directly with tune numbers.
   - Selecting on dt1 only is the D-043(a) design, so this is not a defect. Action: keep, and record the dt4 cost of dt1 selection when reporting dt4.
3. **Low: one transient spike at epoch 3.** Both streams jump at once (v2e dt1 0.567, real 0.582, flow_mag 1.28), then recover by epoch 4. Action: none.
4. **Info: the sanity numbers are plausible.**
   - At epoch 26, real tune dt1 EPE is 0.2995 and v2e is 0.2339 (real is 28% worse). At dt4, real is 0.871 and v2e is 0.709.
   - The partial real_sup run reaches a best real dt1 of 0.222 by epoch 22. That puts the sim-to-real cost at about 0.08 px, which is believable.
   - Zero flow on 200 real tune dt1 windows gives 0.892 (results/baselines/cm_tune.json, zero_flow_reference, same region and mask), so the model is about 3× better than zero flow.
   - These values sit at or below the lower edge of the EV-FlowNet-class range of 0.3–1.0 px. That is reasonable for an in-sequence tune slice (outdoor_day2).
   - Caveat: the v2e and real EPEs use masks built from each stream's own events, so they cover different pixel sets, and the number of scored windows per stream is not logged.
   - Supervised arm, so collapse checks do not apply. The diagnostics are stable anyway: flow magnitude on events is about 1.05, the non-empty IWE ratio about 0.75, and |div| about 0.04.
5. **Low: small provenance gaps.**
   - `n_events` is null for every store.
   - The `.v2e.json` sidecars (which hold the CT and profile) are not hashed; only the store `.json` manifests are.
   - The CUDA RNG is not saved in last.pt. This does not matter here because the run never resumed, but it does matter for real_sup's resume.
   - Action: fix in the loader for future runs. No rerun is needed.
6. **Optional:** to confirm independently that the GPU fault left no silent damage, re-evaluate best.pt in a fresh process on v2e tune dt1. It should reproduce 0.2339 to within bf16 / cudnn.benchmark noise.

**Recommendation: keep.** Use best.pt (epoch 26) as the v2e_fixed_sup voxel5 s1337 cell.
