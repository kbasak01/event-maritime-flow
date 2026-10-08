# SOURCES — event-maritime-flow

Checked 2026-10-08. "Verified" = read on the primary page that day. **(verify)** = could not be
confirmed (page blocked or silent) and must be resolved in Phase 0 into `docs/LICENSES.md`.
All Phase 0 (verify) items were resolved on 2026-10-08; `docs/LICENSES.md` is authoritative.

## Datasets

| Name | Role | Key facts used in the plan | Licence | Source |
|---|---|---|---|---|
| **MVSEC** (Zhu et al., RA-L 2018) | T1 + T2 core | Two DAVIS 346B (346×260) with APS frames + DVS events + IMU; HDF5 files mirror ROS topics (`davis/left/events`, `davis/left/image_raw`, `image_raw_ts`, `image_raw_event_inds`); GT flow `*_gt_flow_dist.npz` with keys `timestamps`, `x_flow_dist`, `y_flow_dist` (from EV-FlowNet, RSS 2018) | CC BY-SA 4.0 (verified, dataset page) | daniilidis-group.github.io/mvsec |
| **DSEC / DSEC-Flow** (Gehrig et al., RA-L 2021; E-RAFT 3DV 2021) | T1 generalisation | 640×480 events; `events/{p,x,y,t}`, `ms_to_idx`, `t_offset`, `rectify_map.h5`; flow GT 16-bit PNG decoded `(v − 2¹⁵)/128`, valid = channel 3; flow at 10 Hz over 100 ms windows; 18 train sequences have flow GT; test GT hidden behind an online benchmark | code MIT; data CC BY-SA 4.0 (resolved 2026-10-08, see `docs/LICENSES.md`) | github.com/uzh-rpg/DSEC, dsec.ifi.uzh.ch |
| **LaRS** (Žust, Perš, Kristan, ICCV 2023) | maritime T3 + class slices + clutter study | 4000+ labelled keyframes, 9 preceding frames each; semantic PNG ids 0 obstacle / 1 water / 2 sky / 255 ignore; panoptic COCO; 19 scene attributes; downloads: images 966 MB, annotations 22.3 MB (train+val only), sequences 13.6 GB | CC BY-NC 4.0, no redistribution (resolved 2026-10-08, see `docs/LICENSES.md`) | lojzezust.github.io/lars-dataset |
| **Singapore Maritime Dataset (SMD)** (Prasad et al., T-ITS 2017) | maritime T3 (long on-board clips) | Canon 70D, 1920×1080; 40 visible on-shore (3 GB), 11 visible on-board (768 MB), 30 NIR on-shore; conditions incl. haze, rain, sunrise/sunset | none stated → research use, no redistribution, no README imagery (resolved 2026-10-08, see `docs/LICENSES.md`) | sites.google.com/site/dilipprasad/home/singapore-maritime-dataset |
| UZH-FPV (Delmerico et al., ICRA 2019) | optional extra self-supervised training data | mDAVIS 346 events; used by tudelft/event_flow for self-supervised training | CC BY-NC-SA 3.0 (resolved 2026-10-08, see `docs/LICENSES.md`) | fpv.ifi.uzh.ch |

Correction to the original project brief: MaSTr1325 is a set of single labelled images, not
sequences, so it cannot feed v2e; LaRS (keyframe + 9 context frames) and SMD (full videos) replace it.
MODS is not used (access is by request and adds little over LaRS for this purpose).

## Simulators and code

| Name | Use | Facts used | Licence | Source |
|---|---|---|---|---|
| **v2e** (Hu, Liu, Delbruck, CVPRW 2021) | event synthesis | conda Python 3.10 install; SuperSloMo39.ckpt (151 MB) needed for interpolation; flags `--dvs346`, `--output_width/height`, `--pos_thres/--neg_thres/--sigma_thres`, `--cutoff_hz`, `--leak_rate_hz`, `--shot_noise_rate_hz`, `--dvs_emulator_seed`, `--auto_timestamp_resolution`, `--timestamp_resolution`, `--input_frame_rate`, `--dvs_h5`, `--no_preview`, `--skip_video_output`, `--stop_time`; README warns 50–200× slower than real time on a low-end GPU at 346×260 and much slower at fine timestamp resolution; documents the per-frame timestamp "pyramid" artefact | MIT (verified) | github.com/SensorsINI/v2e |
| **E-RAFT** | reference model R1 + MVSEC reproduction gate | checkpoints `dsec.tar`, `mvsec_20.tar`, `mvsec_45.tar`; DSEC loader uses 15-bin voxel grids normalised over non-zero cells, 100 ms windows; pre-processed MVSEC `outdoor_day1` 20/45 Hz | MIT (verified) | github.com/uzh-rpg/E-RAFT |
| **tudelft/event_flow** (Hagenaars, Paredes-Vallés, de Croon, NeurIPS 2021) | metric oracle (FWL, RSAT, AEE) vendored into `tests/oracles/` | self-supervised event flow; MVSEC/ECD/HQF evaluation; FWL and RSAT for GT-free data | MIT (verified) | github.com/tudelft/event_flow |
| tub-rip/event_based_optical_flow (Shiba et al., ECCV 2022 / T-PAMI 2024) | optional external CM reference, separate env, outputs only | MultiCM; MVSEC/DSEC tested | **GPL-3.0 (verified) — do not vendor/import** | github.com/tub-rip/event_based_optical_flow |
| torchvision RAFT | T3a teacher | `raft_large` pretrained weights | BSD-3 (torchvision) | pytorch.org/vision |
| rpg_vid2e / ESIM | stretch simulator ablation | — | check before use | github.com/uzh-rpg/rpg_vid2e |

## Papers that shape design decisions

- Stoffregen, Scheerlinck, Scaramuzza, Drummond, Barnes, Kleeman, Mahony, "Reducing the Sim-to-Real
  Gap for Event Cameras," ECCV 2020 (arXiv 2003.09078) — contrast threshold is the key simulation
  parameter; apparent CT of real cameras varies widely; introduces FWL. → §0(4), §1.6.
- Zhu, Yuan, Chaney, Daniilidis, "EV-FlowNet," RSS 2018 — MVSEC GT flow; U-Net baseline. → M1.
- Gehrig, Millhäusler, Gehrig, Scaramuzza, "E-RAFT," 3DV 2021 — DSEC-Flow; R1.
- Shiba, Aoki, Gallego, "Secrets of Event-Based Optical Flow," ECCV 2022; "Event Collapse in Contrast
  Maximization Frameworks," Sensors 2022 — dt = 1 ≈ 22 ms outdoor (45 Hz), dt = 4 ≈ 89 ms; GT absent
  over parts of the image; collapse guard. → §1.4, §1.6.
- Paper noting non-standard `outdoor_day1` evaluation windows in the literature (arXiv 2409.14611). →
  §1.3.
- "Real-Time Event-Based Tracking and Detection for Maritime Environments" (arXiv 2202.04231) — waves
  generate many times more events than vessels (DAVIS346red). → §0(5).
- eStonefish-Scenes (arXiv 2505.13309) and UEOF (arXiv 2601.10054, WACVW 2026) — synthetic underwater
  event-flow datasets; closest prior art on synthetic-event flow in water. → positioning.
- Hérissé, Hamel, Mahony, Russotto, "Landing a VTOL UAV on a moving platform using optical flow,"
  T-RO 2012; Pijnacker Hordijk, Scheper, de Croon, event-based optical-flow landing (arXiv
  1702.00061). → §5.

## Things the original brief stated that this plan does not repeat

- "IROS 2025 event-vision workshop flags missing datasets" — not verified; not cited.
- "MaSTr sequences" — MaSTr1325 has no sequences (see correction above).
