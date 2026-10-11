training-run-diagnostician, 2026-10-11, run runs/train/v2e_rand_sup_voxel5_s2026 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_rand_sup_voxel5_s2026/

**Findings**

1. Manifest: no problems. I diffed this manifest against the s1337 one, which is already TRUSTWORTHY. The only differences are seed, run_id and created_at.
   - git 7904922, dirty=false, launched from the evflow-wt-phase5b worktree. Both config SHAs match the files.
   - The train and tune split SHAs (bdf68aae…, 9ea94aea…) match splits/SHA256SUMS.txt.
   - corpus_plan SHA 254a141e… matches data/converted/v2e/mvsec_d035/corpus_plan.json.
   - All 18 store-manifest SHAs match the .json files on disk. For all 16 rand{0..3} × clip0..3 draws, the recorded seed (2000–2303) and pos/neg/sigma thresholds match those files.
   - end_clamped_us is 115 for rand1 and rand2 train_clip2. Window counts are 10452/10440, as D-048(a) requires.
   - Action: keep.

2. Curves: no problems. All 40 epochs are logged, with no NaN or Inf in any column.
   - Train loss falls from 2.275 to 0.516, and mean grad norm from 3.8 to 0.99.
   - LR follows the cosine from 3e-4 to 0 and is identical to s1337 at every epoch.
   - Predicted flow magnitude stays flat at about 0.76–1.0 px, so there is no runaway.
   - v2e tune dt1 EPE is noisy in epochs 3–6, which is normal early in training, then settles around 0.234–0.244.
   - Action: keep.

3. Selection: correct. The argmin of v2e_dt1_epe over the metrics is epoch 28 (0.23444). This matches selection.json, the final selected_so_far value and the last line of the unit log.
   - sha256 of best.pt is 3608f2b5…, matching selection.json.
   - The only tune data is mvsec_tune (outdoor_day2 frames 3651–6275), which does not overlap the train clips (7588 onward). No test sequence (indoor_flying1–3, outdoor_day1) appears.
   - Action: keep.

4. Real vs v2e tune gap (info): at epoch 28, real dt1 EPE is 0.297 against 0.234 on v2e (about +27%). dt4 is 0.836 against 0.707. s1337 shows the same gap (0.301 / 0.232).
   - real_sup selects at 0.216 (s1337) and 0.218 (s2026), so this arm's real-tune EPE is about 0.08 px higher. That is the expected sim-to-real gap, not a defect.
   - Action: report it as a diagnostic only. Real-tune numbers must not be used for any choice.

5. Seed consistency (info): this seed selects at 0.2344, against 0.2323 for s1337, both at epoch 28. Matching fixed-CT runs select at 0.2339 (s1337) and 0.2344 (s2026).
   - The CT randomisation brings no measurable tune benefit at this sample size.
   - Action: report it neutrally.

6. Process (info): job.log shows the unit running alone, 23:31–01:01, straight after v2e_fixed_sup s2026, with one attempt and no resume.
