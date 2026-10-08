"""Canonical HDF5 event store (IMPLEMENTATION_PLAN §1.2, docs/CONVENTIONS.md) and its slicer.

Layout (DSEC-like, one format for every source):

    events/x, events/y   uint16   pixel coordinates (space given by attr ``distortion``)
    events/t             int64    µs, non-decreasing, relative to ``t_offset``
    events/p             int8     0 = OFF, 1 = ON
    ms_to_idx            int64    DSEC definition, see :func:`compute_ms_to_idx`
    t_offset             int64    µs to add to ``t`` to recover sensor time
    rectify_map          float32  optional (H, W, 2): raw (x, y) -> rectified (x, y) (DSEC)
    attrs: width, height, source, sensor, distortion, polarity_convention
           [, simulator_manifest_sha256 for synthetic data]

``EventSlicer`` follows uzh-rpg/DSEC ``scripts/utils/eventslicer.py`` semantics: times passed in and
returned are absolute (``t_offset`` added), windows are half-open ``t_start <= t < t_end``, and a
window that cannot be fully covered by ``ms_to_idx`` returns ``None``.
"""

from __future__ import annotations

import math
from os import PathLike
from typing import Any

import h5py
import hdf5plugin
import numpy as np

POLARITY_CONVENTION = "0=OFF,1=ON"
DISTORTIONS = ("raw", "rectified", "undistorted")
REQUIRED_ATTRS = ("width", "height", "source", "sensor", "distortion", "polarity_convention")
_CHUNK = 1 << 16


def compute_ms_to_idx(t: np.ndarray) -> np.ndarray:
    """DSEC ``ms_to_idx`` for event times ``t`` (int µs, relative, non-decreasing).

    ``ms_to_idx[ms]`` is the first index with ``t >= ms*1000``, so ``t[ms_to_idx[ms]-1] < ms*1000``.
    Length is ``t[-1]//1000 + 2``: the final entry (ms after the last event's ms) equals N, as in
    the DSEC files (verified on thun_00_a: 11902 entries, last = N).
    """
    t = np.asarray(t)
    if t.size and np.any(np.diff(t) < 0):
        raise ValueError("event times must be non-decreasing")
    n_ms = int(t[-1]) // 1000 + 2 if t.size else 1
    return np.searchsorted(t, np.arange(n_ms, dtype=np.int64) * 1000, side="left").astype(np.int64)


def _check_attrs(width: int, height: int, source: str, sensor: str, distortion: str) -> None:
    if distortion not in DISTORTIONS:
        raise ValueError(f"distortion must be one of {DISTORTIONS}, got {distortion!r}")
    if width <= 0 or height <= 0 or width > 65535 or height > 65535:
        raise ValueError(f"bad sensor size {width}x{height}")
    if not source or not sensor:
        raise ValueError("source and sensor attrs must be non-empty")


class StoreWriter:
    """Streaming writer: ``append`` validated chunks, ``ms_to_idx`` is built incrementally."""

    def __init__(
        self,
        path: str | PathLike,
        *,
        width: int,
        height: int,
        source: str,
        sensor: str,
        distortion: str,
        t_offset: int,
        rectify_map: np.ndarray | None = None,
        extra_attrs: dict[str, Any] | None = None,
        compression: str = "blosc",
    ) -> None:
        _check_attrs(width, height, source, sensor, distortion)
        if rectify_map is not None:
            rectify_map = np.asarray(rectify_map)
            if rectify_map.shape != (height, width, 2):
                raise ValueError(
                    f"rectify_map must be ({height}, {width}, 2), got {rectify_map.shape}")
        self.width, self.height = int(width), int(height)
        self._f = h5py.File(path, "w")
        comp = hdf5plugin.Blosc(cname="lz4", clevel=5) if compression == "blosc" else {}
        g = self._f.create_group("events")
        self._ds = {
            k: g.create_dataset(k, shape=(0,), maxshape=(None,), dtype=dt, chunks=(_CHUNK,), **comp)
            for k, dt in (("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.int8))
        }
        self._f.create_dataset("t_offset", data=np.int64(t_offset))
        if rectify_map is not None:
            self._f.create_dataset("rectify_map", data=rectify_map.astype(np.float32))
        attrs = dict(width=self.width, height=self.height, source=source, sensor=sensor,
                     distortion=distortion, polarity_convention=POLARITY_CONVENTION)
        attrs.update(extra_attrs or {})
        self._f.attrs.update(attrs)
        self._n = 0
        self._last_t: int | None = None
        self._ms_to_idx: list[np.ndarray] = []
        self._next_ms = 0  # first ms whose index is not yet known

    def append(self, *, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray) -> None:
        x, y, t, p = (np.asarray(a) for a in (x, y, t, p))
        n = t.shape[0]
        if not (x.shape == y.shape == p.shape == (n,)):
            raise ValueError("x, y, t, p must be 1-D and the same length")
        if n == 0:
            return
        if not np.issubdtype(t.dtype, np.integer):
            raise ValueError("t must be integer microseconds")
        if t[0] < 0:
            raise ValueError("t must be non-negative (relative to t_offset)")
        if np.any(np.diff(t) < 0) or (self._last_t is not None and t[0] < self._last_t):
            raise ValueError("event times must be non-decreasing")
        if not np.all((p == 0) | (p == 1)):
            raise ValueError("polarity must be in {0, 1}")
        if x.min() < 0 or x.max() >= self.width:
            raise ValueError("x out of range")
        if y.min() < 0 or y.max() >= self.height:
            raise ValueError("y out of range")

        # ms_to_idx entries resolved inside this chunk: every ms with ms*1000 <= t[-1].
        last_ms = int(t[-1]) // 1000
        if last_ms >= self._next_ms:
            ms = np.arange(self._next_ms, last_ms + 1, dtype=np.int64)
            self._ms_to_idx.append(self._n + np.searchsorted(t, ms * 1000, side="left"))
            self._next_ms = last_ms + 1

        for k, a, dt in (("x", x, np.uint16), ("y", y, np.uint16), ("t", t, np.int64),
                         ("p", p, np.int8)):
            ds = self._ds[k]
            ds.resize((self._n + n,))
            ds[self._n :] = a.astype(dt, copy=False)
        self._n += n
        self._last_t = int(t[-1])

    def close(self) -> None:
        if self._f is None:
            return
        # Remaining ms up to last_ms + 1 point past the end (DSEC: final entry == N).
        last_ms = (self._last_t // 1000 + 1) if self._last_t is not None else 0
        if last_ms >= self._next_ms:
            self._ms_to_idx.append(np.full(last_ms + 1 - self._next_ms, self._n, dtype=np.int64))
        m = np.concatenate(self._ms_to_idx) if self._ms_to_idx else np.zeros(1, np.int64)
        self._f.create_dataset("ms_to_idx", data=m.astype(np.int64))
        self._f.close()
        self._f = None

    def abort(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> StoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_store(path: str | PathLike, *, x, y, t, p, **kwargs) -> None:
    """Write a whole event stream at once (see :class:`StoreWriter` for keyword arguments)."""
    with StoreWriter(path, **kwargs) as w:
        w.append(x=x, y=y, t=t, p=p)


class EventStore:
    """Read-only handle on a canonical store."""

    def __init__(self, path: str | PathLike) -> None:
        self.path = path
        self._f = h5py.File(path, "r")
        missing = [a for a in REQUIRED_ATTRS if a not in self._f.attrs]
        if missing:
            raise ValueError(f"{path}: missing store attrs {missing}")
        self.attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                      for k, v in self._f.attrs.items()}
        self.width = int(self.attrs["width"])
        self.height = int(self.attrs["height"])
        self.t_offset = int(self._f["t_offset"][()])
        self.ms_to_idx = np.asarray(self._f["ms_to_idx"], dtype=np.int64)
        self.events = {k: self._f[f"events/{k}"] for k in ("x", "y", "t", "p")}
        self.n_events = int(self.events["t"].shape[0])
        self.has_rectify_map = "rectify_map" in self._f
        self._rectify_map: np.ndarray | None = None

    @property
    def rectify_map(self) -> np.ndarray:
        if not self.has_rectify_map:
            raise ValueError(f"{self.path}: store has no rectify_map")
        if self._rectify_map is None:
            self._rectify_map = np.asarray(self._f["rectify_map"], dtype=np.float32)
        return self._rectify_map

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class EventSlicer:
    """Half-open time-window access to a store, DSEC semantics (absolute µs in and out)."""

    def __init__(self, store: EventStore | str | PathLike) -> None:
        self._own = not isinstance(store, EventStore)
        self.store = EventStore(store) if self._own else store

    @property
    def t_start_us(self) -> int:
        return self.store.t_offset

    @property
    def t_final_us(self) -> int:
        n = self.store.n_events
        return self.store.t_offset + (int(self.store.events["t"][n - 1]) if n else 0)

    def get_events(
        self, t_start_us: int, t_end_us: int, *, rectify: bool | None = None
    ) -> dict[str, np.ndarray] | None:
        """Events with ``t_start_us <= t < t_end_us`` (absolute µs), or ``None`` if the window is
        not fully covered by the store. With rectification x/y are float32 rectified coords from
        ``rectify_map`` (DSEC: ``rectify_map[y, x]``). ``rectify=None`` rectifies exactly when the
        store has a ``rectify_map`` (its GT lives in the rectified frame, D-017); raw coordinates
        of such a store need an explicit ``rectify=False``."""
        if rectify is None:
            rectify = self.store.has_rectify_map
        if not t_start_us < t_end_us:
            raise ValueError("need t_start_us < t_end_us")
        if rectify and not self.store.has_rectify_map:
            raise ValueError("rectify=True but the store has no rectify_map")
        t0 = int(t_start_us) - self.store.t_offset
        t1 = int(t_end_us) - self.store.t_offset
        if t0 < 0:
            return None
        ms0, ms1 = math.floor(t0 / 1000), math.ceil(t1 / 1000)
        m = self.store.ms_to_idx
        if ms1 >= m.size:
            return None
        lo, hi = int(m[ms0]), int(m[ms1])
        t = np.asarray(self.store.events["t"][lo:hi])
        a = lo + int(np.searchsorted(t, t0, side="left"))
        b = lo + int(np.searchsorted(t, t1, side="left"))
        out = {"t": t[a - lo : b - lo] + self.store.t_offset}
        for k in ("x", "y", "p"):
            out[k] = np.asarray(self.store.events[k][a:b])
        if rectify:
            xy = self.store.rectify_map[out["y"].astype(np.intp), out["x"].astype(np.intp)]
            out["x"] = xy[:, 0].astype(np.float32)
            out["y"] = xy[:, 1].astype(np.float32)
        return out

    def close(self) -> None:
        if self._own:
            self.store.close()

    def __enter__(self) -> EventSlicer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
