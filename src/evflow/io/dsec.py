"""DSEC reader/converter (layout verified against the files, docs/DECISIONS.md D-016, D-017).

Source layout (``<seq>/``):
    events_left/events.h5      events/{p uint8, t uint32, x uint16, y uint16} (Blosc),
                               ms_to_idx uint64, t_offset int64 (µs)
    events_left/rectify_map.h5 rectify_map (480, 640, 2) float32, raw (x, y) -> rectified (x, y)
    image_timestamps.txt       absolute µs, one per image
    flow/forward_timestamps.txt  "from_us, to_us" rows (absolute µs)
    flow/forward/NNNNNN.png    16-bit flow PNG named by the image index of from_us

The canonical store keeps raw x/y plus ``rectify_map`` (D-017); rectify on read via
``EventSlicer.get_events(..., rectify=True)``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter used by DSEC files)
import numpy as np

from evflow.flow.png import read_dsec_flow_png
from evflow.io.store import StoreWriter

WIDTH, HEIGHT = 640, 480
SENSOR = "Prophesee Gen3.1 (DSEC left event camera)"


def convert_events(
    seq_dir: str | PathLike, dst: str | PathLike, *, sequence: str, camera: str = "left",
    chunk: int = 10_000_000,
) -> dict:
    """Copy ``events_<camera>`` into a canonical store; the recomputed ``ms_to_idx`` must equal
    the source's exactly."""
    ev_dir = Path(seq_dir) / f"events_{camera}"
    with h5py.File(ev_dir / "rectify_map.h5", "r") as f:
        rmap = np.asarray(f["rectify_map"], np.float32)
    with h5py.File(ev_dir / "events.h5", "r") as f:
        n = f["events/t"].shape[0]
        src_ms = np.asarray(f["ms_to_idx"], dtype=np.int64)
        t_offset = int(f["t_offset"][()])
        n_on = 0
        extra = {"source_file": f"events_{camera}/events.h5",
                 "rectify_map_source": f"events_{camera}/rectify_map.h5",
                 "time_conversion": "uint32 us -> int64 us, t_offset copied",
                 "polarity_mapping": "0->0, 1->1 (uint8 -> int8)"}
        with StoreWriter(dst, width=WIDTH, height=HEIGHT, source=f"dsec/{sequence}/events_{camera}",
                         sensor=SENSOR, distortion="raw", t_offset=t_offset, rectify_map=rmap,
                         extra_attrs=extra) as w:
            for lo in range(0, n, chunk):
                p = f["events/p"][lo : lo + chunk]
                n_on += int(np.count_nonzero(p == 1))
                w.append(x=f["events/x"][lo : lo + chunk], y=f["events/y"][lo : lo + chunk],
                         t=f["events/t"][lo : lo + chunk].astype(np.int64), p=p.astype(np.int8))
            t_last = int(f["events/t"][n - 1])
    with h5py.File(dst, "r") as out:
        ours = np.asarray(out["ms_to_idx"], dtype=np.int64)
    if not np.array_equal(ours, src_ms):
        bad = (np.flatnonzero(ours != src_ms[: len(ours)])[:5] if len(ours) == len(src_ms)
               else f"length {len(ours)} vs {len(src_ms)}")
        Path(dst).unlink()
        raise ValueError(f"recomputed ms_to_idx differs from the source file at {bad}")
    return {"n_events": n, "n_on": n_on, "t_offset_us": t_offset, "duration_us": t_last,
            "width": WIDTH, "height": HEIGHT, "ms_to_idx_matches_source": True}


def read_flow_timestamps(seq_dir: str | PathLike, direction: str = "forward") -> np.ndarray:
    """(N, 2) int64 absolute µs [from, to] per flow GT map."""
    path = Path(seq_dir) / "flow" / f"{direction}_timestamps.txt"
    return np.loadtxt(path, delimiter=",", comments="#", dtype=np.int64, ndmin=2)


class FlowGT:
    """Forward flow GT of one sequence: ``gt[i] -> (flow (2, H, W), valid (H, W), (from, to))``."""

    def __init__(self, seq_dir: str | PathLike, direction: str = "forward") -> None:
        seq_dir = Path(seq_dir)
        self.timestamps_us = read_flow_timestamps(seq_dir, direction)
        image_ts = np.loadtxt(seq_dir / "image_timestamps.txt", dtype=np.int64, ndmin=1)
        idx = np.searchsorted(image_ts, self.timestamps_us[:, 0])
        if np.any(idx >= len(image_ts)) or np.any(image_ts[idx] != self.timestamps_us[:, 0]):
            raise ValueError(f"{seq_dir}: flow from-timestamps do not match image timestamps")
        self.png_paths = [seq_dir / "flow" / direction / f"{i:06d}.png" for i in idx]
        missing = [p for p in self.png_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing flow PNGs, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.png_paths)

    def __getitem__(self, i: int):
        flow, valid = read_dsec_flow_png(self.png_paths[i])
        a, b = self.timestamps_us[i]
        return flow, valid, (int(a), int(b))

