"""Tiny files with the *source* layouts observed in D-015 (MVSEC) and D-016 (DSEC)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

MVSEC_T0_S = 1504645177.425404  # first event time of indoor_flying1 (D-015)


def write_fake_mvsec(path: Path, *, n: int = 5000, width: int = 346, height: int = 260,
                     seed: int = 0, duration_s: float = 0.35) -> dict:
    """``davis/left/events`` (N, 4) float64 [x, y, t (Unix s), p in {-1, +1}] + APS timestamps."""
    rng = np.random.default_rng(seed)
    t = MVSEC_T0_S + np.sort(rng.uniform(0, duration_s, n))
    t[0] = MVSEC_T0_S
    ev = np.stack([
        rng.integers(0, width, n).astype(np.float64),
        rng.integers(0, height, n).astype(np.float64),
        t,
        rng.choice([-1.0, 1.0], n),
    ], axis=1)
    aps_ts = MVSEC_T0_S + 0.02 + np.arange(10) * 0.0319
    with h5py.File(path, "w") as f:
        f.create_dataset("davis/left/events", data=ev)
        f.create_dataset("davis/left/image_raw_ts", data=aps_ts)
        f.create_dataset("davis/left/image_raw_event_inds", data=np.searchsorted(t, aps_ts))
    return {"events": ev, "aps_ts": aps_ts}


def write_fake_dsec(seq_dir: Path, *, n: int = 20_000, width: int = 640, height: int = 480,
                    seed: int = 0, t_offset: int = 49_599_200_165) -> dict:
    """``events_left/events.h5`` (DSEC dtypes) + ``rectify_map.h5`` + flow PNG index files."""
    from evflow.flow.png import write_dsec_flow_png

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 450_000, n)).astype(np.uint32)
    t[0] = 0
    x = rng.integers(0, width, n).astype(np.uint16)
    y = rng.integers(0, height, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8)
    t64 = t.astype(np.int64)
    ms = np.arange(int(t64[-1]) // 1000 + 2, dtype=np.int64) * 1000
    ms_to_idx = np.searchsorted(t64, ms, side="left").astype(np.uint64)
    ev_dir = seq_dir / "events_left"
    ev_dir.mkdir(parents=True)
    with h5py.File(ev_dir / "events.h5", "w") as f:
        for k, a in (("p", p), ("t", t), ("x", x), ("y", y)):
            f.create_dataset(f"events/{k}", data=a)
        f.create_dataset("ms_to_idx", data=ms_to_idx)
        f.create_dataset("t_offset", data=np.int64(t_offset))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    rmap = np.stack([xs * 1.01 - 3.0, ys * 0.99 + 1.5], axis=-1).astype(np.float32)
    with h5py.File(ev_dir / "rectify_map.h5", "w") as f:
        f.create_dataset("rectify_map", data=rmap)

    img_ts = t_offset + 500 + np.arange(10, dtype=np.int64) * 50_000
    np.savetxt(seq_dir / "image_timestamps.txt", img_ts, fmt="%d")
    (seq_dir / "flow" / "forward").mkdir(parents=True)
    rows = [(img_ts[i], img_ts[i + 2]) for i in (2, 4, 6)]
    with open(seq_dir / "flow" / "forward_timestamps.txt", "w") as fh:
        fh.write("# from_timestamp_us, to_timestamp_us\n")
        for a, b in rows:
            fh.write(f"{a}, {b}\n")
    flows = []
    for k, i in enumerate((2, 4, 6)):
        flow = np.zeros((2, height, width))
        flow[0], flow[1] = 1.5 * (k + 1), -0.25
        valid = np.zeros((height, width), bool)
        valid[10:20, 30:50] = True
        write_dsec_flow_png(seq_dir / "flow" / "forward" / f"{i:06d}.png", flow, valid)
        flows.append((flow, valid))
    return {"t": t, "x": x, "y": y, "p": p, "ms_to_idx": ms_to_idx, "rectify_map": rmap,
            "t_offset": t_offset, "flow_rows": rows, "flows": flows}


def write_fake_lars(root: Path, subset: str = "val", *, prefix: str = "fake_seq_01",
                    first: int = 3, n_frames: int = 12, keyframe_local: int = 9,
                    size: tuple[int, int] = (64, 48), seed: int = 0) -> dict:
    """LaRS layout: ``images_seq`` JPGs, one keyframe with a semantic mask (0 obstacle, 1 water,
    2 sky, 255 ignore), ``image_annotations.json`` and ``panoptic_annotations.json``."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    w, h = size
    d = root / subset
    (d / "images_seq").mkdir(parents=True)
    (d / "semantic_masks").mkdir()
    for n in range(first, first + n_frames):
        im = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(im).save(d / "images_seq" / f"{prefix}_{n:05d}.jpg")
    kf = f"{prefix}_{first + keyframe_local:05d}"
    lab = np.full((h, w), 1, np.uint8)            # water below
    lab[: h // 3] = 2                              # sky on top
    lab[h // 3: h // 3 + 4, : w // 2] = 0          # an obstacle band
    lab[-2:, -3:] = 255                            # ignore corner
    Image.fromarray(lab).save(d / "semantic_masks" / f"{kf}.png")
    ann = {"info": {}, "licenses": [], "annotations": [{
        "id": 1, "file_name": f"{kf}.jpg",
        "labels": {"scene_type": "sea_like", "lighting": "day_like", "reflections": None,
                   "waves": "moderate",
                   "special": {"extra_dark": False, "extra_bright": False, "glitter": True,
                               "dirty_lens": False, "wakes": False, "rain": False,
                               "fog": False, "plants_debris": False}}}]}
    (d / "image_annotations.json").write_text(json.dumps(ann))
    pan = {"info": {}, "images": [], "licenses": [],
           "categories": [{"id": 1, "name": "Static Obstacle", "supercategory": "obstacle"},
                          {"id": 3, "name": "Water", "supercategory": "water"},
                          {"id": 5, "name": "Sky", "supercategory": "sky"}],
           "annotations": [{"file_name": f"{kf}.png", "segments_info": [
               {"id": 1, "category_id": 1, "area": int((lab == 0).sum())},
               {"id": 3, "category_id": 3, "area": int((lab == 1).sum())},
               {"id": 5, "category_id": 5, "area": int((lab == 2).sum())}]}]}
    (d / "panoptic_annotations.json").write_text(json.dumps(pan))
    clip = {"clip": f"{prefix}_{first:05d}", "prefix": prefix, "first": first,
            "last": first + n_frames - 1, "n_frames": n_frames, "keyframes": [kf]}
    return {"clip": clip, "label": lab, "keyframe": kf}


def write_fake_smd(path: Path, *, n_frames: int = 20, fps: int = 30,
                   size: tuple[int, int] = (64, 36)) -> dict:
    """A short MJPG AVI whose frame k has mean intensity ~ 10 k (frame identity survives
    lossy coding)."""
    import cv2

    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for k in range(n_frames):
        vw.write(np.full((h, w, 3), 10 * k, np.uint8))
    vw.release()
    return {"n_frames": n_frames, "fps": fps, "size": size}
