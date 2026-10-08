"""Tiny files with the *source* layouts observed in D-015 (MVSEC) and D-016 (DSEC)."""

from __future__ import annotations

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
