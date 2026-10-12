# training-run-diagnostician — maritime_selfsup_voxel5_s1337 (Phase 7)

Verbatim report of the read-only agent (2026-10-12); saved by Claude because the agent may not write files.

VERDICT: TRUSTWORTHY WITH CAVEATS

Run: runs/train/maritime_selfsup_voxel5_s1337/ (CPU-only checks).

**1. Gradient spikes: gone in this run, and the floor is what removes them (pass, with one note).** grad_norm_max peaks at 7.27 (epoch 26), grad_norm_mean 0.18–0.45; Phase 5 v2e_selfsup peaked at 1.25e6 with means up to 2332. Clipped steps 99 of 5840 (1.7 %), highest epoch 4.8 % (epoch 36); no NaN/Inf. Same-weights test (12 batches, floor on vs off): Phase 5 checkpoint (epoch 22) on v2e MVSEC train windows — without the floor gradient norm median 0.41, max 67.2; with min_count 0.01 median 0.077, max 0.119. This run's checkpoint on its maritime windows: no spikes with the floor off either (max 0.85 vs 0.92). So the floor removes the D-049 spikes, but the maritime data would not have triggered them; this run alone does not prove the fix. The floor also changes the objective's value (epoch-38 data term on 192 centre-cropped train windows 0.612 with, 0.652 without): it is part of what this arm optimises (D-056).

**2. Train loss does not fall after epoch 0 (MEDIUM; keep).** train_loss 0.618 (epoch 0) → 0.639 (epoch 19) → 0.635; cm_data flat at 0.614–0.620. Smoothness term grows 0.022 → 0.099 as flow magnitude grows 0.40 → 1.39 px (plateau from ≈ epoch 27). The model did learn the objective (data term 0.693 at zero flow vs 0.612 at epoch 38 on the same 192 windows), almost all within epoch 0; the late epochs drift towards larger flow with no gain in the objective.

**3. Event collapse: none during training (pass).** v2e-tune diagnostics epoch 0 → 38: var_ratio 0.70 → 0.78 while nonempty_ratio 1.03 → 0.98 (collapse direction, ≈ 5 %, ending at v2e_fixed_sup's level 0.778 / 1.05). abs_div 0.061–0.077, not growing, under D-048's 0.1; ≈ 2× Phase 5 v2e_selfsup (0.028) and maritime_teacher (0.028), above v2e_fixed_sup (0.042). Final tune flow_mag / median over epochs 0.91. FWL and EPE move together: Spearman(v2e FWL, real-tune EPE) across epochs −0.75; epoch 38 is also the best real-tune EPE epoch.

**4. Logged real-tune EPE is worse than zero flow in all 40 epochs (MEDIUM, for interpreting results).** Epoch 38: real dt1 EPE 1.317, v2e 1.228 (logged only). Zero flow 0.892 (200-window reference, `results/baselines/cm_tune.json`), M0b 1.004, Phase 5 v2e_selfsup 0.756, maritime_teacher 0.777. Real flow magnitude 1.76 vs ≈ 0.85 for real_sup: ≈ 2× over-prediction on driving content. Domain transfer, not a training defect; FWL 1.145 here is not accuracy, and the T3/T2 rows and gate should say so.

**5. Selection (pass).** argmax v2e_dt1_fwl at epoch 38 (1.14498) = selection.json; best.pt epoch 38, sha256 5965c697…0e06d = selection.json, DONE and summary. Epochs 36–38 differ by 0.002 FWL (within noise).

**6. No leakage (pass).** 118 training stores: 8 smd_train + 110 lars_train, all ×1.0; trial units left out. Tune stores: v2e outdoor_day2 tune store and real outdoor_day2 store, clip 3651–6275. smd_test, lars_eval, mvsec_test absent from the manifest; load_maritime raises LeakageError outside smd_train/lars_train or CT ≠ 1.

**7. Manifest (pass, minor note).** git b552569 clean; config SHAs match b552569 and HEAD; seed 1337; CM weights 0.1 / 1.0, min_count 0.01; mvsec_tune split SHA = SHA256SUMS; maritime split SHAs bound through the corpus plan (82f9f8ad…, recomputed, matching); every store has store_sha256 (spot-check lars_mastr_0576_00000 OK); end_clamped_us for the 4 D-060 stores. Cosine LR 3e-4 → 0, 3493 s, no resume; attempt 1 left only a manifest. Minor: train_epe logged 0, not NaN, for an arm without GT; launch worktree now at 5eea7f9, manifest SHA is the authority.

**Actions:** keep the run; record findings 2 and 4 as caveats in the gate; before any future v2e_selfsup re-run with the floor, note in DECISIONS that the floor changes the gradient size ≈ 5× on MVSEC v2e data (finding 1).
