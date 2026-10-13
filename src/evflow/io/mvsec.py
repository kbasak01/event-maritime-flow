"""MVSEC reader/converter (layout verified against the files, docs/DECISIONS.md D-015).

Source layout (``<seq>_data.hdf5``):
    davis/<cam>/events        (N, 4) float64 columns [x, y, t, p], t Unix seconds, p in {-1, +1}
    davis/<cam>/image_raw_ts  (F,) float64 Unix seconds
GT (``<seq>_gt_flow_dist.npz``, uncompressed members, 20 Hz): ``timestamps`` (F_gt,) Unix s,
``x_flow_dist`` / ``y_flow_dist`` (F_gt, 260, 346) float64, px displacement from timestamps[i] to
timestamps[i+1], raw (distorted) pixel space = the space of the raw events.

Times leaving this module are int64 microseconds (absolute, ``round(t_s * 1e6)``).
"""

from __future__ import annotations

import zipfile
from os import PathLike
from pathlib import Path

import cv2
import h5py
import numpy as np

from evflow.io.store import StoreWriter

WIDTH, HEIGHT = 346, 260
SENSOR = "DAVIS346B"


def _seconds_to_us(t_s: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(t_s, np.float64) * 1e6).astype(np.int64)


def convert_events(
    src: str | PathLike, dst: str | PathLike, *, sequence: str, cam: str = "left",
    chunk: int = 4_000_000,
) -> dict:
    """Stream ``davis/<cam>/events`` into a canonical store. Raises on any layout surprise
    (non-integer x/y, polarity outside {-1, +1}, decreasing time)."""
    with h5py.File(src, "r") as f:
        ev = f[f"davis/{cam}/events"]
        if ev.ndim != 2 or ev.shape[1] != 4:
            raise ValueError(f"expected (N, 4) events, got {ev.shape}")
        n = ev.shape[0]
        t_offset = int(_seconds_to_us(ev[0, 2]))
        n_on = 0
        with StoreWriter(dst, width=WIDTH, height=HEIGHT, source=f"mvsec/{sequence}/davis/{cam}",
                         sensor=SENSOR, distortion="raw", t_offset=t_offset,
                         extra_attrs={"source_file": Path(src).name,
                                      "time_conversion": "round(t_s*1e6) - t_offset",
                                      "polarity_mapping": "-1->0, +1->1"}) as w:
            for lo in range(0, n, chunk):
                b = ev[lo : lo + chunk]
                x, y, t_s, p = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                if np.any(x != np.floor(x)) or np.any(y != np.floor(y)):
                    raise ValueError(f"non-integer x/y in events[{lo}:{lo + len(b)}]")
                if not np.all((p == 1.0) | (p == -1.0)):
                    raise ValueError(f"polarity outside {{-1, +1}} in events[{lo}:{lo + len(b)}]")
                pol = (p > 0).astype(np.int8)
                n_on += int(pol.sum())
                w.append(x=x.astype(np.uint16), y=y.astype(np.uint16),
                         t=_seconds_to_us(t_s) - t_offset, p=pol)
        t_last = int(_seconds_to_us(ev[n - 1, 2])) - t_offset
    return {"n_events": n, "n_on": n_on, "t_offset_us": t_offset, "duration_us": t_last,
            "width": WIDTH, "height": HEIGHT}


def read_aps_timestamps_us(src: str | PathLike, cam: str = "left") -> np.ndarray:
    with h5py.File(src, "r") as f:
        return _seconds_to_us(f[f"davis/{cam}/image_raw_ts"][:])


class GTFlow:
    """Lazy access to ``*_gt_flow_dist.npz`` (memory-mapped when members are stored)."""

    def __init__(self, path: str | PathLike) -> None:
        self.path = Path(path)
        with np.load(self.path) as z:
            self.timestamps_s = np.asarray(z["timestamps"], np.float64)
        self.timestamps_us = _seconds_to_us(self.timestamps_s)
        self._x = self._open("x_flow_dist")
        self._y = self._open("y_flow_dist")
        if self._x.shape != self._y.shape or self._x.shape[0] != len(self.timestamps_s):
            raise ValueError(f"{path}: inconsistent GT shapes {self._x.shape} {self._y.shape}")
        self.shape = self._x.shape[1:]

    @property
    def memmapped(self) -> bool:
        return isinstance(self._x, np.memmap)

    def _open(self, key: str) -> np.ndarray:
        with zipfile.ZipFile(self.path) as zf:
            info = zf.getinfo(f"{key}.npy")
            if info.compress_type != zipfile.ZIP_STORED:
                with np.load(self.path) as z:
                    return np.asarray(z[key])
        with open(self.path, "rb") as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            if local[:4] != b"PK\x03\x04":
                raise ValueError(f"{self.path}: bad zip local header for {key}")
            name_len = int.from_bytes(local[26:28], "little")
            extra_len = int.from_bytes(local[28:30], "little")
            fh.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(fh)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = reader(fh)
            offset = fh.tell()
        return np.memmap(self.path, dtype=dtype, mode="r", offset=offset, shape=shape,
                         order="F" if fortran else "C")

    def __len__(self) -> int:
        return len(self.timestamps_s)

    def field(self, i: int) -> np.ndarray:
        """GT field i, (2, H, W) float64: displacement from timestamps[i] to timestamps[i+1]."""
        return np.stack([np.asarray(self._x[i], np.float64), np.asarray(self._y[i], np.float64)])


def _interval_schedule(ts: np.ndarray, t_start_us: int, t_end_us: int) -> list[tuple[int, float]]:
    """(GT interval index, fraction of that interval to advance) for a window that spans at least
    one full GT interval. The first entry covers the rest of the start interval; every later
    interval up to and including the first one that reaches ``t_end_us`` follows, fully (1.0) or
    in part. There is always a last entry, so a window of exactly one aligned GT interval ends
    with a fraction of 0: that field is still looked up, which only updates the validity masks
    (D-019, D-068)."""
    i = int(np.searchsorted(ts, t_start_us, side="right")) - 1
    sched = [(i, float((ts[i + 1] - t_start_us) / float(ts[i + 1] - ts[i])))]
    while True:
        i += 1
        if ts[i + 1] >= t_end_us:
            sched.append((i, float((t_end_us - ts[i]) / float(ts[i + 1] - ts[i]))))
            return sched
        sched.append((i, 1.0))


def _advect(gt: GTFlow, schedule: list[tuple[int, float]]) -> np.ndarray:
    """Displacement (2, H, W) of every pixel carried through the scheduled fields. Each step
    looks the field up at the pixel nearest to the current float32 position (cv2.remap
    INTER_NEAREST, constant 0 outside the image); a component that ever reads exactly 0 is
    invalid and returned as 0."""
    h, w = gt.shape
    pos = np.mgrid[0:h, 0:w].astype(np.float32)[::-1].copy()   # (x, y) positions
    start = pos.copy()
    alive = np.ones((2, h, w), bool)
    for i, frac in schedule:
        field = gt.field(i)
        look = np.stack([cv2.remap(field[c], pos[0], pos[1], cv2.INTER_NEAREST)
                         for c in (0, 1)])
        alive &= look != 0
        pos += (look * frac).astype(np.float32)
    return np.where(alive, pos - start, 0.0).astype(np.float64)


def gt_flow_for_window(gt: GTFlow, t_start_us: int, t_end_us: int) -> tuple[np.ndarray, np.ndarray]:
    """GT flow (2, H, W) over [t_start, t_end] (absolute µs) and its validity (H, W).

    Follows the propagation rule described in EV-FlowNet (RSS 2018; written from the paper's
    description and configs/eval_mvsec.yaml, no code from that repository, D-019, D-068): a
    window shorter than its GT interval takes that field scaled linearly; a longer window
    carries every pixel through the overlapped fields (``_interval_schedule``) with
    nearest-pixel lookup (``_advect``). Valid = finite and |flow| > 0.
    """
    ts = gt.timestamps_us
    if not (ts[0] <= t_start_us < t_end_us <= ts[-1]):
        raise ValueError(f"window [{t_start_us}, {t_end_us}] outside GT span [{ts[0]}, {ts[-1]}]")
    it = int(np.searchsorted(ts, t_start_us, side="right")) - 1
    gt_dt = float(ts[it + 1] - ts[it])
    dt = float(t_end_us - t_start_us)
    if gt_dt > dt:
        flow = gt.field(it) * (dt / gt_dt)
    else:
        flow = _advect(gt, _interval_schedule(ts, t_start_us, t_end_us))
    valid = np.all(np.isfinite(flow), axis=0) & (np.hypot(flow[0], flow[1]) > 0)
    return flow, valid
