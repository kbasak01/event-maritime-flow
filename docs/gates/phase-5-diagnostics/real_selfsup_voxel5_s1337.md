training-run-diagnostician, 2026-10-10, run runs/train/real_selfsup_voxel5_s1337 (verbatim)

VERDICT: TRUSTWORTHY

I found no event collapse over the 40 epochs, and selection, checkpoint and splits all check out. I did not start CUDA or import torch.

**Manifest: PASS**
- Run directory: runs/train/real_selfsup_voxel5_s1337/
- git 7904922, not dirty. The run was launched from worktree ../evflow-wt-phase5b, whose HEAD is the same commit. Seed 1337.
- Both config SHAs match the worktree files.
- Resolved config: `losses.cm.objective: average_timestamp`, smoothness_weight 0.5, divergence_penalty 1.0, reference times 0/0.5/1.
- Split SHAs bdf68aae (train) and 9ea94aea (tune) match `splits/SHA256SUMS.txt` and the two real_sup runs.
- The only event store is outdoor_day2, tune clip [3651, 6275]. No test split appears in the manifest.

**Curves: PASS**
- 40 rows, no NaN/Inf.
- Train loss falls from 0.8185 to about 0.790. It plateaus after epoch 20, which is expected for an average-timestamp loss.
- LR follows per-step cosine to 0 exactly (max deviation 5e-20).
- The unit log is one session (161 lines, epochs 0 to 39, no resume).

**Grad-norm spikes: Minor**
- grad_norm_max reaches 10,340 (epoch 30) and 8,903 (epoch 37); per-epoch mean goes up to 16.
- grad_clip is 1.0 and the loss is not disturbed. Action: keep, mention in the gate file.

**Selection: PASS**
- Argmax of real_dt1_fwl is epoch 23 (1.08937), matching selection.json.
- The top five epochs are within 0.0006 of each other, so the pick is rule-correct but within noise. Logged EPE at epochs 22 to 39 is 0.73 to 0.78, so the choice barely matters.
- best.pt sha256 7990a808…665e matches selection.json and DONE. Its mtime (18:42) matches the epoch-23 log line (18:42:21).

**Event collapse: none, full weight**

| Signal (tune dt1) | This run | Limit / reference |
|---|---|---|
| flow_mag, epochs 9 to 39 | 1.13 to 1.33, no growth | final/median 1.00, limit 2x |
| IWE non-empty ratio | 0.96 to 0.98 | ≥ 0.8; real_sup 0.97 |
| IWE variance ratio | 0.72 rising to 0.82 | real_sup 0.89; no rise-while-emptying |
| mean \|div\| | about 0.012 | ≤ 0.1; real_sup 0.044 to 0.049 |

- The collapsed D-047 cell hit flow_mag 4.4, variance ratio 3.1, \|div\| 0.21 and EPE 4.07 within 2 epochs.
- Epoch 3 had a brief non-empty dip to 0.859 that recovered by epoch 4.
- Magnitude bias (Minor): averaged over the region, flow is larger than real_sup's (1.20 vs 0.85). On event pixels it is smaller (1.35 vs 1.67). So the run overestimates where there are no events and underestimates where there are, which is a typical self-supervised scale bias. Action: note when interpreting results.

**Sanity: PASS, with a caveat**
- Logged GT EPE at epoch 23 is 0.774 (below zero-flow 0.892). The supervised runs reach 0.216 / 0.218. FWL 1.089 is above real_sup (1.06) and M0b (1.083).
- Caveat (Minor): real_dt1_epe covers all 2,614 tune windows, but the zero-flow 0.892 comes from the 200 M0b windows in results/baselines/cm_tune.json. The comparison is indicative only. Action: if this comparison is reported, compute zero flow on the same windows.

**Provenance: Minor**
- The CM weights reach the run only through CLI flags (`--cm-smoothness 0.5 --cm-divergence 1.0`). They are not in configs/train_real_selfsup.yaml.
- The file that justifies them, ../evflow-wt-phase5b/results/train/cm_sweep.json, is untracked and exists only in the worktree, not in the main repo. Its chosen cell sw0.5_dp1 (FWL 1.0880) matches this run.
- Action: bring cm_sweep.json into the main repo's results/ with the run's results, or put the weights into the arm config, before the gate.

**Recommendation:** keep the run.
