"""v2e ``--dvs_h5`` reader/converter (layout verified on a v2e output, docs/DECISIONS.md D-022).

Observed layout (v2e @ configs/v2e_commit.txt, ``v2ecore/emulator.py``): one dataset ``events``
(N, 4) **uint32**, gzip, columns **[t, x, y, p]**; t = µs from the first source frame
(v2e computes ``uint32(float32(t_s) * 1e6)``, so its precision is the float32 ulp of t in seconds:
≤ 3.8 µs below 64 s); p ∈ {0, 1} with 1 = ON (v2e maps its −1 OFF to 0; sign checked against APS
log-intensity change, D-022). No frame datasets are written in the frame-folder/video path.

The converter maps t through an optional ``time_map`` (v2e's uniform clock → real sensor clock,
``sim/v2e_runner.make_clock_map``), stable-sorts if the mapped times are not non-decreasing, and
writes the canonical store (``io/store.py``) with ``simulator_manifest_sha256``.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike

import h5py
import numpy as np

from evflow.io.store import StoreWriter

COLUMNS = ("t", "x", "y", "p")


def read_v2e_events(src: str | PathLike) -> dict[str, np.ndarray]:
    """All events of a v2e h5 as {t int64 µs, x uint16, y uint16, p int8}; raises on surprises."""
    with h5py.File(src, "r") as f:
        if "events" not in f:
            raise ValueError(f"{src}: no 'events' dataset (keys {list(f)})")
        ev = f["events"]
        if ev.ndim != 2 or ev.shape[1] != 4 or ev.dtype != np.uint32:
            raise ValueError(f"{src}: expected (N, 4) uint32 events, got {ev.shape} {ev.dtype}")
        a = ev[:]
    p = a[:, 3]
    if not np.all((p == 0) | (p == 1)):
        raise ValueError(f"{src}: polarity outside {{0, 1}}")
    return {"t": a[:, 0].astype(np.int64), "x": a[:, 1].astype(np.uint16),
            "y": a[:, 2].astype(np.uint16), "p": p.astype(np.int8)}


def _iter_chunks(src: str | PathLike, chunk: int):
    with h5py.File(src, "r") as f:
        ev = f["events"]
        if ev.ndim != 2 or ev.shape[1] != 4 or ev.dtype != np.uint32:
            raise ValueError(f"{src}: expected (N, 4) uint32 events, got {ev.shape} {ev.dtype}")
        for lo in range(0, ev.shape[0], chunk):
            a = ev[lo : lo + chunk]
            p = a[:, 3]
            if not np.all((p == 0) | (p == 1)):
                raise ValueError(f"{src}: polarity outside {{0, 1}}")
            yield {"t": a[:, 0].astype(np.int64), "x": a[:, 1].astype(np.uint16),
                   "y": a[:, 2].astype(np.uint16), "p": p.astype(np.int8)}


def convert_v2e_h5(
    src: str | PathLike, dst: str | PathLike, *, width: int, height: int, source: str,
    sensor: str, t_offset_us: int = 0, time_map: Callable[[np.ndarray], np.ndarray] | None = None,
    time_map_domain_end_us: float | None = None, simulator_manifest_sha256: str,
    chunk: int = 4_000_000,
) -> dict:
    """Write the canonical store for one v2e output. ``t_offset_us`` is the absolute time of the
    first source frame (MVSEC: its APS timestamp); stored t = mapped v2e t (≥ 0).

    Streams in chunks; if the (mapped) times are not non-decreasing it falls back to loading all
    events and stable-sorting them. ``time_map_domain_end_us`` (v2e time of the last clock knot)
    lets it count events the map would clamp."""
    attrs = {"simulator": "v2e", "simulator_manifest_sha256": simulator_manifest_sha256,
             "time_conversion": "clock-mapped v2e t" if time_map else "v2e t as-is"}
    kw = dict(width=width, height=height, source=source, sensor=sensor, distortion="raw",
              t_offset=int(t_offset_us), extra_attrs=attrs)
    stats = {"n": 0, "n_on": 0, "beyond": 0, "raw_nonmono": 0, "last_raw": None, "last": 0}

    def mapped(e: dict) -> np.ndarray:
        if e["x"].size and (e["x"].max() >= width or e["y"].max() >= height):
            raise ValueError(f"{src}: events outside {width}x{height}")
        t = np.asarray(time_map(e["t"]) if time_map is not None else e["t"], np.int64)
        if t.size and t[0] < 0:
            raise ValueError("mapped times before the first frame")
        return t

    sorted_fallback = False
    try:
        with StoreWriter(dst, **kw) as w:
            for e in _iter_chunks(src, chunk):
                raw = e["t"]
                if raw.size:
                    prev = stats["last_raw"]
                    stats["raw_nonmono"] += int(np.count_nonzero(np.diff(raw) < 0)) + int(
                        prev is not None and raw[0] < prev)
                    stats["last_raw"] = int(raw[-1])
                if time_map_domain_end_us is not None:
                    stats["beyond"] += int(np.count_nonzero(raw > time_map_domain_end_us))
                t = mapped(e)
                w.append(x=e["x"], y=e["y"], t=t, p=e["p"])   # raises if not non-decreasing
                stats["n"] += t.size
                stats["n_on"] += int(e["p"].sum())
                if t.size:
                    stats["last"] = int(t[-1])
    except ValueError as err:
        if "non-decreasing" not in str(err):
            raise
        sorted_fallback = True
    nonmono = 0
    if sorted_fallback:
        ev = read_v2e_events(src)
        t = mapped(ev)
        nonmono = int(np.count_nonzero(np.diff(t) < 0))
        order = np.argsort(t, kind="stable")
        t = t[order]
        with StoreWriter(dst, **kw) as w:
            for lo in range(0, t.size, chunk):
                idx = order[lo : lo + chunk]
                w.append(x=ev["x"][idx], y=ev["y"][idx], t=t[lo : lo + chunk], p=ev["p"][idx])
        stats.update(n=int(t.size), n_on=int(ev["p"].sum()), last=int(t[-1]) if t.size else 0,
                     raw_nonmono=int(np.count_nonzero(np.diff(ev["t"]) < 0)))
        if time_map_domain_end_us is not None:
            stats["beyond"] = int(np.count_nonzero(ev["t"] > time_map_domain_end_us))
    n, dur = stats["n"], stats["last"]
    return {"n_events": n, "n_on": stats["n_on"],
            "on_fraction": stats["n_on"] / n if n else None,
            "duration_us": dur, "events_per_s": n / (dur / 1e6) if dur else None,
            "t_offset_us": int(t_offset_us), "nonmonotonic_raw": stats["raw_nonmono"],
            "nonmonotonic_after_map_sorted": nonmono, "sorted_fallback": sorted_fallback,
            "n_events_beyond_clock_map": (stats["beyond"] if time_map_domain_end_us is not None
                                          else None),
            "width": width, "height": height}
