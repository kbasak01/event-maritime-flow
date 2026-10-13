# Commit map (history rewrite, D-069)

On 2026-10-13 the history was rewritten with `git filter-repo` (v2.47.0) to replace the commit author/committer email with the GitHub noreply address and to make absolute home paths in old documents repo-relative. Every commit SHA changed; trees of the final commit are identical. Results JSON, gate records, decisions, run manifests and the release's `run_manifests.json` still cite the **old** SHAs (their contents were not rewritten): resolve them here. Short SHAs in commit messages were rewritten by filter-repo.

| # | date | old SHA | new SHA | subject |
|---|---|---|---|---|
| 1 | 2026-10-08 | `acfce0cc31dbae72754e663e7856f327e90e2a0f` | `61d3023570a783381b7f53db6e2bef06df7c0a04` | Starter kit: plan, Claude Code agents and skills |
| 2 | 2026-10-08 | `4f103f75b4083bcee4e6847eada67ce9e5b90732` | `630b00ec19ba7a2d93960a7d5b06694e4a1e3b14` | Phase 0: package skeleton, tests, data inventory, licences, envs |
| 3 | 2026-10-08 | `023060e82f5344d2f0f94b5cf8b10d8b55e767a6` | `d948b9ddeda77e6e96ac6f624490b8b9c581ff75` | Phase 0 gate: record green CI run, mark phase 0 PASS |
| 4 | 2026-10-08 | `4a15461640a517636190790492239ec72aaf02dd` | `e9ec45ef4972966976deacc866aac1129a41c35b` | Data acquisition: fetch_data.py, data_sources.yaml, data/ root |
| 5 | 2026-10-08 | `36c7ccd5d0f617c2db46db0351dc2c38b1440095` | `6ffc46a6b9f994240b9f3432c932c7f7aea19455` | Data: outdoor_day2_data.hdf5 in place; inventory complete |
| 6 | 2026-10-08 | `1d1b78253803a9b85ef87bd433bce8714104f880` | `941ed682cdec2235a36778bc99eac8918e7630c1` | Phase 1: event core — store, slicer, representations, metrics, IWE/FWL, converters |
| 7 | 2026-10-08 | `d25b2283ca08936c50e1fb255666d2261ce9aa57` | `31b230d23682e90455678e5da6c7aebe25494760` | Phase 1 gate: verifier + auditor PASS, record verdicts, tick A1/A2/A4/A5/C1/C2 |
| 8 | 2026-10-08 | `ab557bceb253a2f73fb6274a310248c74844cf0f` | `ce2b87e73440578cc55add53f3b20b58c5c703b6` | Phase 1 gate: record green CI run, mark phase 1 PASS |
| 9 | 2026-10-08 | `fbc216dba37659fc3c18eda9af01033cd085b643` | `61e8cb12db240886214e61632dcef60bf9ea753b` | Phase 2: v2e runner, event statistics, calibration, splits, throughput budget |
| 10 | 2026-10-08 | `79bddba0fb05787a910cdd45c4930b0b24c7c22b` | `a0331d4a1d0f322d9a3c9eb1537047d6c8a9000f` | Phase 2: calibration + throughput results; freeze mvsec_cal, mvsec_rand, NCC threshold |
| 11 | 2026-10-08 | `d4d1cd22fc6a17d6932db773c5e549ee5ad3b5ea` | `ddf3c4b27a420c1e246a8a2f2f0eed2e5a48077e` | Phase 2 audit fixes: budget counts spent sweep, maritime cuts, provenance fields, doc corrections |
| 12 | 2026-10-08 | `fe1860e090f8438a89b82e39a49b527dba0ea768` | `fe54a0cfd502cb0f7e9f56ff28b241fbdce8ce90` | Phase 2: regenerate throughput and LaRS fps JSON from d4d1cd2 (tier n/a, measured sweep) |
| 13 | 2026-10-09 | `6c5d2cc9da977f9f8fdefd0a2ea3a20a6eb0746b` | `d130ee485e5c9632b97bfa3a1f28a9451ed6dcfc` | Phase 2 re-audit fixes: reject wrapping x/y, tests for clamp guard and git_dirty, D-026/D-031 notes |
| 14 | 2026-10-09 | `4f28991d6fa584f6d0316aa4590a79a8b024b172` | `a5241fe74f6b6cea1ffe1fdb08a1d6cdf7709bf6` | Phase 2 gate: PASS (both auditors); trials budgeted, D-026/D-031 wording, checklist A3 A6 B1 B2 B3 B6 |
| 15 | 2026-10-09 | `7921940cd7dc103e06bc0a2de85868ad700377c4` | `52d38e82ea5a54ab089218bdade8f54920f395ac` | Phase 2: regenerate throughput JSON (trial lines, LaRS 110) from gate commit |
| 16 | 2026-10-09 | `662a037f93af4bf02f833fe3c6a920fb34d94398` | `eac5462e6ab3fb6af974c0f9f3b22266c80ea2c1` | Phase 2 gate: record green CI run, mark phase 2 PASS |
| 17 | 2026-10-09 | `97210ab8cc43c3f01b411fdf895b2d8e173a1b32` | `d7b5a5fce9ff27438a6015359a8a2163981594fe` | Phase 3: MVSEC v2e corpus plan, runner split guard, job v2e-mvsec |
| 18 | 2026-10-09 | `a8a0f16be7f513b1db7c7c73c475c903356587e0` | `ed3d32349da551c2808d90e166a3c68c655ad1fa` | Phase 3: fidelity report and count-image alignment check (D-033, D-034) |
| 19 | 2026-10-09 | `c737c09df8eb9715a4c585c16f77a4ccedce368f` | `b37fb1292fb1992441511d6c8e0ba572ede081c7` | Phase 3: corpus done (30 units, clean commit), fidelity report, alignment check, budget re-cost |
| 20 | 2026-10-09 | `a20474c4ea2187c80612b6967743c1d4a3211d9a` | `c3161c48033e7f79f91029dda9f209349ea2ab8e` | Phase 3: regenerate results JSON from clean commit c737c09 (metadata only) |
| 21 | 2026-10-09 | `547bd34ca85e96e467e868ef012b1526469510d2` | `c8b86029a8eb66cbf573f1b3870072b65daf1eed` | Phase 3 gate: FAIL (v2e ~0.5-frame timing lag from the 60 Hz lowpass); lag profile in alignment check |
| 22 | 2026-10-09 | `81d1cddd46a3458904cef979355ae52d50e9ed3d` | `02b726f1b1f9746960fc76cc1307dc3784734dcb` | Phase 3 D-035: timing-constrained re-calibration grid on mvsec_tune, job prepared |
| 23 | 2026-10-09 | `b6853e691de1b9546dac0e77774301660393ea89` | `ff423ea8bc50ce61354070bd792ea2c36069a4ab` | Phase 3 D-035: freeze timing-constrained profile (t10: thres 0.6, sigma 0.04, cutoff 0); corpus re-run prepared |
| 24 | 2026-10-09 | `02b975fd432793032a249797849f2c1452fa9c20` | `8a1e593b57b348ad7bb8b913533d1103af1bc391` | Phase 3 D-035 results: corpus re-run (30/30, clean b6853e6), fidelity, alignment with lag profile |
| 25 | 2026-10-09 | `944d8709f1cc6d23ac03834a79ad72fff3a47496` | `e06c69d309ea68ff71a755a05ada34ca66d24c7e` | Phase 3 re-audit fixes: threshold source, LIMITATIONS detail, pause GPU accounting (D-036) |
| 26 | 2026-10-09 | `bd2f89f21df237378ce2713b5d96b9e66aec3e0c` | `747ad368a66da0b5b54c6df7f7102d1ba7e52670` | Phase 3 gate: PASS (both auditors) after D-035; results regenerated at 944d870; B4 B5 |
| 27 | 2026-10-09 | `76a3aae5ba7c2d88a8a95d7301fcfeeac8f7dd4c` | `e120c854b645a8c6d9b4739e7d7b18efbd5b36c0` | Phase 3 gate: record green CI run, mark phase 3 PASS |
| 28 | 2026-10-09 | `0da94b33371f02e1ac2bc37e5f4f974ad94948ad` | `2d146e447a598d72674083f58cea28f4361f5e65` | Phase 4: M0a/M0b baselines, vendored E-RAFT (R1), eval plumbing, reproduction gate, job prepared |
| 29 | 2026-10-10 | `366368415bc9e98e9b8a049ebf52f19f844049f1` | `8ea139f38cb504b4fa0d363b637cade464b867d8` | Phase 4: R1 mvsec_test diagnostic script (pipeline error vs generalisation) |
| 30 | 2026-10-10 | `b9346c5aa6fdd79db633e1b0abc3fda02a2e6449` | `1a282b0bb45d6443e43f681f51318a3a447b124d` | Phase 4: R1 diagnostic warm arm runs real predecessor chains; dt4 and motion-band evidence (verifier finding 1) |
| 31 | 2026-10-10 | `e58e6d9aa28707b4e7201d4907daed1ecd9298a7` | `cab2db1109dfafbe40d70e37a9d6a87d7a211618` | Phase 4 gate: PASS (flow-metric-verifier PASS); baseline results, reproduction gate met |
| 32 | 2026-10-10 | `902dde2e49b6ebb0d78cdefc63763c0a3822cf93` | `89844a702812c072f2f52f2cf8fa7e0d2ce286f3` | Phase 4 gate: record green CI run, mark phase 4 PASS |
| 33 | 2026-10-10 | `a2b1fa005d1cfdd0c2aba10b0b1d07b87797e99c` | `7413f110a38ac47f80b25192ca017c6c6bdd7997` | Phase 5: M1 EV-FlowNet, supervised + CM losses (collapse guard), training loop, arm configs; training job prepared (D-043) |
| 34 | 2026-10-10 | `f1967dec3cee7a47decb33e67025bd8ee5bdb031` | `ca0edbf3f878615df6cd0302670c83559c51a334` | Phase 5: fix GPU resume (RNG state on CPU); job concurrency 1, failed units no longer abort (D-045) |
| 35 | 2026-10-10 | `f9e996e4cc45b98b190638ab38d457131b053aad` | `66bfa287078c7b7d3104394b4abce178eb174443` | Phase 5: stop CM sweep (event collapse, D-047); job SKIP_CM switch, append-mode unit logs |
| 36 | 2026-10-10 | `28f84d4fc3852ae8e67dd6a1b4acfc742908189e` | `ab552298dc8243c7b78391660723e3dd44450e9c` | Phase 5: average-timestamp self-supervised loss (D-047), rand-store end clamp, stricter CM eligibility, one CUDA process at a time (D-048) |
| 37 | 2026-10-10 | `7904922a2883e9b05addaac3a639aa194be8258c` | `5dca015025448cf02637532aac6cbd1bcaf0b422` | Phase 5: AT pilot passes label-free collapse checks (D-048d); status |
| 38 | 2026-10-11 | `fe8ef001e60afa12a967aa658ddf6f478f00e2bf` | `3e611ce2a57644bd72da777f18d719ec2226f81e` | Phase 5: guards from the selection-leakage audit (GT cache vs split manifest/warm-up, sweep freeze checks split and stream, resume recorded in manifest); DECISIONS order and provenance notes |
| 39 | 2026-10-11 | `cb1fb2180d2d7fbf949a7610089e7409fb08ebf5` | `5dc3d4a50353b2cb54756a2fa196dbee1ad493ff` | Phase 5 gate: PASS (17/17 runs TRUSTWORTHY, event-data-auditor PASS); training summary, CM sweep and real_sup gate units |
| 40 | 2026-10-11 | `83f72c5294ed045d1be6e9ceee2ed5d3d843a4c1` | `35d46838d1c1430b61aadb68703ed5af5fce86e6` | Phase 5 gate: record green CI run, mark phase 5 PASS |
| 41 | 2026-10-11 | `8e5aab94c29d48f4cf0ea8fe5dc29d27c291c237` | `d9127d10846584d8d16b35a7aa2e46dab5647dee` | Phase 6: paired bootstrap, T2 (v2e test corpus) eval, summariser, tables, figures; eval-all job prepared (D-051) |
| 42 | 2026-10-11 | `b19e60900023e6cfecb9c7f34a213d9ebbe74c63` | `aac6f0ea50e943f18bd722045cd0057866667ca5` | Phase 6: eval-all attempt 1 stopped (CPU threads); job pins 4 OMP threads, epe_map as the reference sqrt(u^2+v^2) (D-052) |
| 43 | 2026-10-11 | `88cd806a0f81d49e66d8686bcd9f59498b1f0c73` | `fc699a74ef16cf145a8f215c9c1a3c43da8e1228` | Phase 6 gate fixes: seed-pairing ranges and seed-unstable closure (D-053), paired-series provenance, Table B n and reading notes; CI runs make_tables --check |
| 44 | 2026-10-11 | `614e87234888fed3bc98977a8274d8745ce29088` | `dc1f507626759acbca0352b629715280a565b716` | Phase 6 gate: PASS (sim2real-study-runner PASS, eval-reproducibility-auditor REPRODUCED); 298 eval units, Tables A-D, EPE figures |
| 45 | 2026-10-11 | `27bc50b11e8b38573a7cdaf803e2555bfe66fe58` | `a6a7e286e237c577f7b050c0e4ffd014a4e7dcd6` | Phase 6 gate: record green CI run, mark phase 6 PASS |
| 46 | 2026-10-12 | `749480f5b8100e310dfe3a18d643183a5d67bdb8` | `9959d95658cc9f814aff100e1ec0df062850a9df` | Phase 7: maritime readers, v2e corpus, RAFT teacher, maritime training arms (D-054..D-056) |
| 47 | 2026-10-12 | `7fd867c41a4c6b99739417aaf2e7768addc00c2a` | `266e06b6c52edb5650d5a15f72974bb21d75b890` | Phase 7: v2e-maritime job manifest (gpu-job prep) |
| 48 | 2026-10-12 | `a0c832356dadd98b5d457882c6eb2f2a4b0340c4` | `762773f22e8f302f9fab0dc8545ffb2ba8afb410` | Phase 7: T3 evaluation, sea-clutter study, T3 tables, job scripts (D-057, D-058) |
| 49 | 2026-10-12 | `e480fe0431f418600c329e530b6d30c7c7622022` | `ca30549eb9ee8a7081c5511fed465046860160b9` | Phase 7: flow-metric-verifier PASS; minor findings fixed (RSAT common windows, seed alignment, resample counts, tests); limitations |
| 50 | 2026-10-12 | `849bf0a49acf5e8ddca443276fef54fd2e96d483` | `3dde587ca9df4b52724283a9399dcf07223bc8f4` | Phase 7: maritime v2e corpus done (243/243 units, clean at 7fd867c); measured budget 19.44/19 GPU-h (D-059, user decision pending) |
| 51 | 2026-10-12 | `cdc358b789e13e404a7a64dd4a822e8aca58fe93` | `18bdfc74cd5d2b49b6b5b7e91ff223a6966a8339` | Phase 7: teacher-flows job manifest (gpu-job prep) |
| 52 | 2026-10-12 | `b398b765a22152cbd9be50b7ec5ff52582eacd8a` | `d42c3d865c71b585aa5e2ccd853ba621b5a789a7` | Phase 7: accept empty tails of near-static maritime clips up to the known last frame (D-060) |
| 53 | 2026-10-12 | `0037430620a498ad8c181e4fd32e029157c37ca5` | `25669050fd2cd3db015bd55050fbd15fd6f4fc46` | Phase 7: event-data-auditor PASS; text fixes (clip lengths, keyframes, D-059 cause); LaRS-vs-SMD overlap check script |
| 54 | 2026-10-12 | `67e83331d3a0ffcbe162552af9eeaaec424d4544` | `d46c97cdad8274df4a613b7ead69d394091293d4` | Phase 7: LaRS smd* clips vs SMD on-board overlap check: no copies (D-061) |
| 55 | 2026-10-12 | `4e17d5900d4cf279055d25650a8fe7c4a7aa3a2c` | `89bfaa8545f075a2791e3a74d00d250bd30a4354` | Phase 7: user accepted the v2e budget overrun; limit 19.5 GPU-h, measured 19.44 within limit (D-059) |
| 56 | 2026-10-12 | `53dd33d8a6eb6fb192a78b0ead31a5c526b9f27e` | `fc01d44d693d44430eb26003ed075795fd1b5946` | Phase 7: RAFT teacher flows done (161 units; results/teacher/summary.json, D-055) |
| 57 | 2026-10-12 | `4164c3d85e7cd41aa24bde357099382de4f4f762` | `f3313da0312fa6d1163d5ac5d04cdf4889dab356` | Phase 7: train-maritime job manifest (gpu-job prep); STATUS |
| 58 | 2026-10-12 | `b552569bb7377ef3c35fcaa2367598cc79d886fa` | `17238dd3d600b9db9198c15daa45adbdccebb967` | Phase 7: windows of clamped maritime stores checked against the loaded range (D-060 amendment; train attempt 1 failed at epoch 0) |
| 59 | 2026-10-12 | `9df5b5954feac5ffb3a630f6e25b836ec0e4aa49` | `857c0deb3566c6d501962db55ac2c45b70ad0f20` | Phase 7: maritime arms trained (selected epoch 38 each; results/train/maritime_summary.json) |
| 60 | 2026-10-12 | `5eea7f9d9b78a2dcfb83b3f84d086557ee6052b8` | `c944e55d4937e7c6cbbfbeb7aab7d0a929815574` | Phase 7: eval-maritime job manifest (gpu-job prep); STATUS |
| 61 | 2026-10-12 | `3cfe6fa6b7f5e90e5aa445a27473bb8b4ee815dd` | `01fe6bb128f269a7f4f0b86da549f98bf289db23` | Phase 7: training-run-diagnostician TRUSTWORTHY WITH CAVEATS for both maritime runs (D-062) |
| 62 | 2026-10-12 | `0e8537728803bf7487a42db549647a23f7c50e4f` | `4f689cf8669802344a4457249811ffbc04256725` | Phase 7: summarise_maritime block lengths for SMD only; clutter CT ratio over clips with events at x1.0 (n recorded) |
| 63 | 2026-10-12 | `c71bceaf3406034bca7560cbab5729694cc73221` | `8c82ee10e6278a17d9ca1f766786ddb33654fdd8` | Phase 7: T3 evaluation (176/176 units), sea-clutter study, Tables E-H, clutter figure |
| 64 | 2026-10-12 | `9ead16a90766af2d183c64bc3527501325055457` | `da59b0757330d1f743d367d36ec1f9173c095396` | Phase 7: claim-trace fixes in D-054/D-059/D-062 (untraced numbers removed or sourced, estimate corrected to 1.75 GPU-h, tiers stated) |
| 65 | 2026-10-12 | `0ec95f724e82c9f46ce8c07019510f5352e1e50f` | `106dabbe01682f3fea510d7084786cb1514a1806` | Phase 7: sim2real-study-runner findings: Table E zero-shot/one-seed caveat, Table G note, n rows in F/G, summary tier field |
| 66 | 2026-10-12 | `4cd8c714f2ddef8ad7a2e00db6e811dc2004d1a5` | `1d9f6664312b4a7d6029e733d0fcc49fb298c37f` | Phase 7: regenerated T3 summary and tables (clean) |
| 67 | 2026-10-12 | `828b3e4a3bd739459700d77166d4029f304e6785` | `854710dd0f14c3852ac1173042ad95066202e2f4` | Phase 7 gate: PASS (sim2real-study-runner PASS, eval-reproducibility-auditor REPRODUCED); maritime T3 tables, clutter study, checklist E2/E5/E7 |
| 68 | 2026-10-12 | `8b6f61458141bc9b9e99b20fe01a8a4ba26af264` | `262cb36fba627a8a8904a4b358c4126bb43c737a` | CI: install ffmpeg (ffprobe) for the io/smd.py probe test |
| 69 | 2026-10-12 | `14939bc0af7d30114e577b9f2b4f6b55fa52c61c` | `c139a9f40550c638d713705d0efd47258ef8efd6` | Phase 7 gate: record green CI run, mark phase 7 PASS |
| 70 | 2026-10-12 | `15e71432714046f0e64d6dca8d5255ee70839667` | `a8663aaba2c0405c957569f6c70fc031d49f04f2` | Phase 8: ONNX export, parity and latency harness (D-063); export-bench job prepared |
| 71 | 2026-10-12 | `db4f7c359a2a5a671fc1952a1c1e61ec614aaef2` | `8062531eaffe2ec15a366df7420d072b69fdfd4a` | Phase 8: auditor findings F-2, M-1..M-5 (torch fp16 diagnostic, D-063 correction, wording) |
| 72 | 2026-10-12 | `604b0a66ca9d58e46155a52f2b287f392bfa642e` | `299b0b6e69b8c2af60bd611cb1caaca424fd001f` | Phase 8 gate: FAIL (fp16 parity above 5e-2 px; fp32, EPE change, latency pass); results JSON |
| 73 | 2026-10-12 | `486a05194b09188b98a2389c96ae8ebc04893845` | `037cb70c683fbf2712603578fc4885010adadd98` | Phase 8: fp16 sensitivity diagnostic on mvsec_tune (gate F-1 follow-up) |
| 74 | 2026-10-12 | `13e34d8758198c7d13ecc97ea8ef30e216c537f2` | `0544e647b6c06821ea95c55e5f7ee629f88fa995` | Phase 8: D-064 fp16 non-conforming (user decision); gate field, sensitivity table |
| 75 | 2026-10-12 | `5f567c545b9a22286db02c70a25c5900a769ddc6` | `c43a7e54f5aaa8e55ed5878584fa9671e2426bf8` | Phase 8: re-audit findings N-1..N-5 (bf16 variant, wording, appended logs, tier note, gate logic) |
| 76 | 2026-10-12 | `f02904eca26e29ddd4b55d2fb0d66d62a3e2ebd6` | `5d23d8ba5a826eeedd72f5f149c8ad839f9bc46c` | Phase 8 gate: PASS with recorded fp16 non-conformance (D-064); results regenerated at 5f567c5 |
| 77 | 2026-10-12 | `8928598a631b6d170a1277b23684a0bac13abf29` | `67dcd9f9b9b3dd2624b945f6500254a6a8beb5b8` | Phase 8 gate: record green CI run |
| 78 | 2026-10-12 | `24571a6fc88226c04a226998b23da55ea94de747` | `ae995da13f63a9676c8905deabe2a29a70e0e7dc` | Phase 9 gate: PASS (validation sweep; no critical line unmet) |
| 79 | 2026-10-12 | `743df7ef381fdb568dc6ae39e4ce60a812cc4652` | `5248155eb1a14b887a5c6e3b6da70087bf225d27` | Phase 9 gate: record green CI run |
| 80 | 2026-10-12 | `9c76dc4a4fb96a4fab65adae77fb921338202be8` | `9ef3ce58f0212cf05455e4020a341876132815f3` | Phase 10 gate: PASS (gallery README, claim-trace CLEAN, release assets, reviewer SHIP) |
| 81 | 2026-10-12 | `c1265a3d00bd17a63be0ca1a86e378a0cfcaee25` | `7bac0e5bc4bdd766bc37c42cb3738ea3c2b66b1f` | Phase 10 gate: record green CI run |
| 82 | 2026-10-12 | `50743eddc217c46a0ae2ee3aa10f32244197ba3b` | `41d55883fcb787743bc3e7169c05e0f56d862a97` | io/mvsec: restructure GT propagation as interval schedule + advection loop (D-068) |
