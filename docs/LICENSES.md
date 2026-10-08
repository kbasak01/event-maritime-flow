# LICENSES

Filled in Phase 0 from SOURCES.md. Every row was checked on the primary page on 2026-10-08. "No raw
data" means no frames, event files, GT or v2e outputs of the dataset are committed or released;
code, metrics and aggregate numbers derived from the data are.

## Datasets

| Item | Kind | Licence / terms | Redistribution allowed? | README imagery allowed? | Attribution text | Checked on |
|---|---|---|---|---|---|---|
| MVSEC | data | CC BY-SA 4.0 ("This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License", daniilidis-group.github.io/mvsec) | No raw data in repo/release (size + policy). Derived figures: share-alike, CC BY-SA 4.0 | Yes: attributed, marked CC BY-SA 4.0 | Zhu, Thakur, Özaslan, Pfrommer, Kumar, Daniilidis, "The Multi Vehicle Stereo Event Camera Dataset," IEEE RA-L 3(3):2032–2039, 2018. GT flow: Zhu, Yuan, Chaney, Daniilidis, "EV-FlowNet," RSS 2018 | 2026-10-08 |
| DSEC — data | data | CC BY-SA 4.0 (dsec.ifi.uzh.ch: "Creative Commons Attribution-ShareAlike 4.0 International public license"; commercial use allowed) | No raw data in repo/release. Derived figures: share-alike, CC BY-SA 4.0 | Yes: attributed, marked CC BY-SA 4.0 | Gehrig, Aarents, Gehrig, Scaramuzza, "DSEC: A Stereo Event Camera Dataset for Driving Scenarios," IEEE RA-L 2021; flow: Gehrig, Millhäusler, Gehrig, Scaramuzza, "E-RAFT," 3DV 2021 | 2026-10-08 |
| DSEC — code (uzh-rpg/DSEC) | code (reference only) | MIT (GitHub repo licence) | Yes, if vendored: keep header (none vendored in Phase 0) | n/a | as above | 2026-10-08 |
| LaRS | data | CC BY-NC 4.0 ("All images and annotations are released under the CC-BY-NC 4.0 license", lojzezust.github.io/lars-dataset/terms.html). Research use for maritime obstacle detection; no commercial use; datasheet: lojzezust.github.io/lars-dataset/static/pdfs/lars_datasheet.pdf | **No.** "Direct sharing of the images or annotations with third parties is strictly prohibited"; refer others to the dataset page. Algorithm results may be shared if the original data cannot be derived from them | Yes: the terms allow images in publications/presentations. Attributed, marked CC BY-NC 4.0, non-commercial | Žust, Perš, Kristan, "LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark," ICCV 2023 | 2026-10-08 |
| LaRS evaluator (lojzezust/lars_evaluator) | code (reference only) | No licence file in the repo → all rights reserved by default | Not vendored, not copied; only invoked as an external tool if needed | n/a | cite the LaRS paper | 2026-10-08 |
| SMD (Singapore Maritime Dataset) | data | No licence stated on the dataset page. The page's acknowledgement reads: "Dataset has been captured by Dilip K. Prasad and annotated by student volunteers." Captured personally under varied conditions (40 min before sunrise, sunrise, midday, afternoon, evening, 2 h after sunset, haze, rain), July 2015 – May 2016 | **No.** No licence = no redistribution right. Use for research only | **No** unless the author grants written permission (then attributed) | Prasad, Rajan, Rachmawati, Rajabally, Quek, "Video Processing from Electro-optical Sensors for Object Detection and Tracking in Maritime Environment: A Survey," IEEE T-ITS 18(8):1993–2016, 2017; plus the acknowledgement sentence above | 2026-10-08 |
| UZH-FPV (optional, not yet used) | data | CC BY-NC-SA 3.0 ("free for non-commercial use (including research)", fpv.ifi.uzh.ch) | No raw data. Derived figures: NC-SA | Only if used: attributed, marked CC BY-NC-SA 3.0 | Delmerico, Cieslewski, Rebecq, Faessler, Scaramuzza, "Are We Ready for Autonomous Drone Racing? The UZH-FPV Drone Racing Dataset," ICRA 2019 | 2026-10-08 |

## Code and models

| Item | Kind | Licence / terms | Redistribution allowed? | README imagery allowed? | Attribution text | Checked on |
|---|---|---|---|---|---|---|
| v2e (SensorsINI/v2e) | code (external venv, not vendored) | MIT (LICENSE: "Copyright (c) 2019 SensorsINI") | n/a, not vendored; pinned commit in `configs/v2e_commit.txt` | n/a | Hu, Liu, Delbruck, "v2e: From Video Frames to Realistic DVS Events," CVPRW 2021 | 2026-10-08 |
| SuperSloMo39.ckpt (v2e interpolation weights) | model (external, not redistributed) | Distributed by v2e's README (Google Drive); no separate licence stated | Not redistributed | n/a | Jiang et al., "Super SloMo," CVPR 2018 (via v2e) | 2026-10-08 |
| E-RAFT (uzh-rpg/E-RAFT) | code (vendored model, Phase 4) | MIT | Yes: keep header and add an entry here when vendored | n/a | Gehrig, Millhäusler, Gehrig, Scaramuzza, "E-RAFT: Dense Optical Flow from Event Cameras," 3DV 2021 | 2026-10-08 (SOURCES.md) |
| tudelft/event_flow | code (**vendored** test oracle: `tests/oracles/event_flow_metrics.py`, commit e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f; `utils/iwe.py` L4–92, `loss/flow.py` L13–23 and L304–628; never imported by `src/`) | MIT (LICENSE: "Copyright (c) 2021 TU Delft", verified in the clone) | Yes: MIT header and notice kept in the file | n/a | Hagenaars, Paredes-Vallés, de Croon, "Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks," NeurIPS 2021 | 2026-10-08 |
| uzh-rpg/DSEC (scripts) | code (reference only, **not vendored**) | MIT (LICENSE: "Copyright (c) 2021 Robotics and Perception Group"), commit af90ef1 | n/a: `ms_to_idx`/slicer semantics and the flow-PNG decode are re-implemented in `src/evflow/io/store.py` and `src/evflow/flow/png.py`; the docstring example is quoted in `tests/test_store.py` | n/a | Gehrig et al., RA-L 2021 | 2026-10-08 |
| daniilidis-group/EV-FlowNet | code (reference only, **not vendored, not imported**) | Penn non-commercial research licence: no redistribution of the software or modifications without Penn's approval (commit cf801ef) | **No**: the MVSEC GT-propagation algorithm (RSS 2018) is re-implemented from its description in `src/evflow/io/mvsec.py`; no code copied (D-019) | n/a | Zhu, Yuan, Chaney, Daniilidis, "EV-FlowNet," RSS 2018 | 2026-10-08 |
| tub-rip/event_based_optical_flow | code | GPL-3.0 | **NOT vendored, NOT imported** (`tests/test_no_gpl_imports.py`); outputs only, from a separate env | n/a | Shiba, Aoki, Gallego, ECCV 2022 (if outputs are used) | 2026-10-08 (SOURCES.md) |
| torchvision RAFT (code) | code (dependency) | BSD-3-Clause (torchvision; the original princeton-vl/RAFT repo is also BSD-3-Clause) | n/a, pip dependency | n/a | Teed, Deng, "RAFT," ECCV 2020 | 2026-10-08 |
| torchvision RAFT weights (`Raft_Large_Weights`) | model (downloaded at runtime) | No separate weight licence on the torchvision docs page. `*_V1` weights were ported from the original paper; `*_V2` were trained from scratch by torchvision. Sintel/KITTI fine-tuned variants inherit those datasets' research-use terms | Not redistributed. The teacher is used only to produce T3a pseudo-labels | n/a | as above | 2026-10-08 |

## Repository

`src/`, `tests/` and `scripts/` are MIT (`LICENSE`). Any vendored third-party file keeps its own
header and gets a row above.
