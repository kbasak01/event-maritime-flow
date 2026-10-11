training-run-diagnostician, 2026-10-10, run runs/train/v2e_selfsup_voxel5_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/v2e_selfsup_voxel5_s1337/

**F1 Gradient spikes (MEDIUM, keep this run; fix before more seeds)**
- Evidence: grad_norm_max (pre-clip) peaks at 1.25e6 at epoch 32, and also hits 7.1e5 at 21, 5.7e5 at 26 and 9.5e4 at 39. The bigger issue is grad_norm_mean. It sits around 3–15 for epochs 0–20, then 82–2332 for epochs 21–39, so the late epochs are mostly heavily clipped steps, not one bad step. All four v2e_selfsup sweep cells show the same pattern (gmax 2e4–8.5e4 over 10 epochs). real_selfsup peaks at about 1e4.
- No damage shows. Loss falls steadily (0.792 → about 0.756) with no jump at 21, 26 or 32 (epoch 32 is 0.7564, 33 is 0.7563). Tune FWL stays flat at 1.139–1.147. Real dt1 EPE keeps improving (0.718 at 32, 0.686 at 33). There are no NaN/Inf values anywhere. Clipping the total norm to 1.0 keeps every update bounded, and the LR at epoch 32 is 2.2e-5.
- Cause is plausible. In `average_timestamp_loss`, `avg = tsum/(cnt+1e-9)` (src/evflow/losses/cm.py:127). Where two events with tiny bilinear weights k1, k2 share a pixel, ∂avg/∂k1 = k2(w1−w2)/(k1+k2)², which grows like 1/k. The 1e-9 term does nothing to regularise this. Those pixels also count toward `hit`. Real events seem to trigger it less, perhaps because their tails overlap less, but that is a guess.
- Action: keep. Before running seeds 2026/7, add a count floor (e.g. `cnt.clamp(min=τ)`, or mask pixels with cnt < τ, τ ≈ 1e-2) and log the fraction of clipped steps. Record this in DECISIONS.

**F2 Event collapse: none (pass)**
- v2e tune diagnostics, epoch 0 → 22 → 39:
  - flow_mag_events: 1.43 → 1.66 → 1.49, bounded.
  - nonempty_ratio: 0.999 → 1.009 → 0.970, no sharp drop.
  - var_ratio: 0.80 → 0.84 → 0.95. It rises, but the nonempty fraction does not fall with it.
  - abs_div: 0.017 → 0.028 → 0.026, below v2e_fixed_sup on the same windows (0.044).
- Real-tune flow_mag is 1.27 at epoch 22, against real_sup at 0.83 and real_selfsup at 1.17. That is the same roughly 1.5× over-prediction seen in the real_selfsup arm. Note it, but it is not collapse.

**F3 Selection (pass)**: argmax of v2e_dt1_fwl over the 40 epochs is epoch 22 (1.14691), matching selection.json. FWL is on a plateau, so the choice among epochs is close to noise (epoch 28 is 1.1470 as well). best.pt sha256 `28148a01…970a` matches selection.json.

**F4 Manifest/config (pass)**
- Present: git 7904922 (clean), seed 1337, split-manifest SHAs (match splits/SHA256SUMS.txt), corpus-plan SHA, and per-store manifest SHAs.
- Config SHAs match both trees.
- Resolved config: objective `average_timestamp`, smoothness 0.1, divergence 1.0, which matches the chosen cell in cm_sweep.json (FWL 1.1413).
- Minor: n_events is null for every store.

**F5 Curves/LR (pass)**: Per-step cosine from 3e-4 to 0, as configured (epoch 20 lr 1.382e-4 is correct). wall_time 8714 s, one session with no resume. train_epe (log only) falls from 2.15 to 1.58.

**F6 No test split (pass)**: The only stores are the train_cal clips, v2e tune and the real outdoor_day2 tune clip 3651–6275. "test" does not appear in loop.py.

**F7 Real vs v2e gap (INFO)**
- At epoch 22: real dt1 EPE 0.756 vs v2e 0.767; dt4 real 1.655 vs v2e 1.522; real dt1 FWL 1.082 vs v2e 1.147.
- Real EPE beats zero flow (0.892, a 200-window reference; indicative only) and matches real_selfsup (0.774 at its epoch 23).
- Tier T2. The real numbers are logged only and played no part in the choice.
