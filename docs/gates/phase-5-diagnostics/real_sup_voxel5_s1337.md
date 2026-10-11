training-run-diagnostician, 2026-10-10, run runs/train/real_sup_voxel5_s1337 (verbatim)

VERDICT: TRUSTWORTHY

Run: runs/train/real_sup_voxel5_s1337/

**Manifest (OK)**
- Config SHAs in manifest.json match the current files: evflow.yaml 3cdc804c…, train_real_sup.yaml 0c93d720…. No config, split or data code changed between a2b1fa0 and f1967de. The f1967de diff touches only `torch.load(map_location="cpu")` in the resume branch, plus the job script.
- Split manifest SHAs match splits/mvsec_train.json (bdf68aae…) and splits/mvsec_tune.json (9ea94aea…). Seed is 1337 and git_dirty is false.
- best.pt sha256 = 0abee62e…697b. This matches selection.json and DONE. The best.pt payload says epoch 36. last.pt says epoch 39, scheduler last_epoch = optimizer step = 26120 (= 40×653).

**Curves and the epoch 23/24 boundary (OK)**
- LR matches 3e-4·½(1+cos(π·(e+1)·653/26120)) exactly (max deviation 0) at every epoch, 23→24 included (1.0365e-4 → 9.2597e-5). The cosine did not restart.
- Train loss keeps its trend: 0.6005, 0.5913 | 0.5793, 0.5668. Mean grad norm: 1.177, 1.172 | 1.191, 1.175. Tune dt1 EPE: 0.2224, 0.2277 | 0.2258, 0.2178. The resumed log's "best 0.2224 @ 22" shows the restored `best` state was correct.
- Every row has 653 steps. There is no NaN/Inf anywhere, and the loop raises an error on non-finite loss or selection metric.
- train_s dropped from about 155 to 71 s after the resume. This is explained by D-045 (concurrency 2→1); the metrics are unaffected.

**No leakage from the crashed epoch 24 (OK)**
- loop.py writes a metrics row and last.pt only after an epoch finishes, including its tune eval. On resume it keeps only CSV rows with epoch < start_epoch.
- metrics.csv has exactly 40 unique epochs, 0–39. The first resume attempt crashed at `set_rng_state`, before any write.
- Per-epoch sampler and worker generators (seed·7919+e, seed·104729+e) set the data order and the crop/flip augmentation (torch RNG in workers). So the resumed epochs drew the same samples as an uninterrupted run would have. Only CUDA nondeterminism differs, as D-045 already says.

**Selection (OK)**
- argmin real_dt1_epe = 0.21602 at epoch 36, which matches selection.json. The `selected_so_far` column is consistent with a running argmin at every row.
- Loaders use `guard_split`, which only allows mvsec_train and mvsec_tune, plus a check on the GT cache split. The manifest lists only mvsec_train and mvsec_tune caches and the outdoor_day2 store. No test split appears.

**Sanity vs references (OK)**
- Tune dt1 EPE is 0.216, against zero flow at 0.892 (CI 0.71–1.07, 200 windows; the window set differs, but the gap is about 4×). The v2e_fixed_sup run reaches 0.30 on real tune dt1, so the in-domain arm is better, as expected.
- Predicted tune flow magnitude (0.82–0.86 px) is close to the zero-flow EPE, so the output scale is plausible. Collapse diagnostics are stable: IWE non-empty ratio about 0.97, mean |div| 0.04–0.055.

**Findings**
1. Low — the manifest does not record the resumed session. manifest.json still shows git a2b1fa0, while epochs 24–39 ran at f1967de, and selection.json has no code SHA. Evidence: runs/phase5-train/job.log line 1 and the f1967de diff. The difference does not affect behaviour. Action: keep. D-045 already documents it; consider writing a resume record (session SHA, start epoch) in future runs.
2. Low — tune EPE plateaus. Values are 0.2160–0.2208 from epoch 25 on and tune dt4 has been flat at about 0.69 since epoch 12, while train EPE keeps falling (0.31→0.27). Epoch 36 beats epoch 26 by only 0.0004 px, which is within noise, so the choice is rule-correct but not meaningful. There is mild overfitting with no degradation. Train EPE mixes dt1 and dt4 with augmentation, so it is not directly comparable to tune. Action: keep. Do not read anything into the exact selected epoch.
3. Info — 0.216 px is below the 0.3–1.0 px literature range. The tune slice is on outdoor_day2, the training sequence, with the center_256_no_hood mask, so it is optimistic and not comparable to literature numbers. The gate is indoor_flying1 (D-043h). Action: report only tier-tagged eval numbers, never this tune value, as accuracy.
4. Info — D-045 says "23 completed epochs saved", but last.pt held epochs 0–23, which is 24. The epoch 35 tune eval also took about 100 s longer than usual (epoch_s 176), with normal values. Action: fix the wording in D-045; nothing else.
