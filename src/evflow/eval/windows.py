"""Evaluation windows and masks for MVSEC (configs/eval_mvsec.yaml, D-019) and DSEC-Flow.

MVSEC: window k = [aps_ts[k], aps_ts[k + dt]] for every APS frame k in the split manifest's
``aps_first .. aps_last`` (stride 1 frame); its predecessor (warm start, E-RAFT's previous
volume) is window k - dt, which ends where window k starts.
DSEC: window i = the i-th forward-flow GT interval [from, to]; its predecessor is i - 1 when
``to[i-1] == from[i]``, else none (the warm-start chain restarts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    index: int                # APS frame k (MVSEC) or GT index i (DSEC)
    t_start_us: int           # absolute µs
    t_end_us: int
    prev_start_us: int | None  # start of the predecessor window (ends at t_start_us), if any


def mvsec_windows(aps_ts_us: np.ndarray, *, aps_first: int, aps_last: int, dt: int) -> list[Window]:
    ts = np.asarray(aps_ts_us, np.int64)
    if not 0 <= aps_first <= aps_last or aps_last + dt >= ts.size:
        raise ValueError(f"APS range [{aps_first}, {aps_last}] + dt={dt} outside {ts.size} frames")
    out = []
    for k in range(aps_first, aps_last + 1):
        prev = int(ts[k - dt]) if k - dt >= 0 else None
        out.append(Window(k, int(ts[k]), int(ts[k + dt]), prev))
    return out


DSEC_SPAN_TOLERANCE_US = 1000  # GT intervals measure 99,930-100,058 µs in dsec_test


def dsec_windows(flow_timestamps_us: np.ndarray, *, delta_t_us: int) -> list[Window]:
    """Windows of DSEC forward-flow GT ([from, to] as listed; spans are 100 ms to within tens of
    µs). The predecessor is the GT window before it when the two are contiguous. R1 reads its
    own volumes as E-RAFT does, [from - delta_t, from) and [from, from + delta_t)."""
    ts = np.asarray(flow_timestamps_us, np.int64)
    out = []
    for i, (a, b) in enumerate(ts):
        if abs(int(b - a) - delta_t_us) > DSEC_SPAN_TOLERANCE_US:
            raise ValueError(f"GT window {i} spans {b - a} µs, expected {delta_t_us}")
        contiguous = i > 0 and int(ts[i - 1, 1]) == int(a)
        out.append(Window(i, int(a), int(b), int(ts[i - 1, 0]) if contiguous else None))
    return out


def region_mask(region: dict, height: int, width: int) -> np.ndarray:
    """Boolean (H, W) mask of a half-open ``{rows: [r0, r1], cols: [c0, c1]}`` region."""
    (r0, r1), (c0, c1) = region["rows"], region["cols"]
    if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
        raise ValueError(f"region {region} outside {height}x{width}")
    m = np.zeros((height, width), bool)
    m[r0:r1, c0:c1] = True
    return m


def events_in_window(x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    """MVSEC sparse mask: pixel has >= 1 event (either polarity) in the window. Integer
    (raw) coordinates only: MVSEC GT and events share the raw distorted pixel grid (D-015)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(y.dtype, np.floating):
        raise ValueError("events_in_window expects integer pixel coordinates")
    m = np.zeros((height, width), bool)
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    m[y[ok].astype(np.intp), x[ok].astype(np.intp)] = True
    return m
