"""Evaluate one unit (model x dataset x sequence x dt) window by window (Phase 4 baselines).

Per window: predicted forward flow (2, H, W) in pixels over the window, the §1.6 metrics from
``flow.metrics.flow_metrics`` under the dataset's masks, and FWL (``interpolation='round'``,
``t_ref = 1``, CONVENTIONS) of the window's events. MVSEC masks follow configs/eval_mvsec.yaml:
valid GT AND events-in-window (sparse) AND the sequence's region; FWL uses the events inside the
region. DSEC: valid GT over the full 640x480 rectified frame, FWL on all events.

Predictors: ``zero`` (M0a), ``cm`` (M0b), ``eraft`` (R1, warm-started along predecessor chains:
the state after window w is handed to the window that starts where w ends).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from evflow.eval.windows import Window, events_in_window, region_mask
from evflow.events.representations import voxel_grid
from evflow.flow.fwl import fwl
from evflow.flow.metrics import flow_metrics
from evflow.models.cm_baseline import CMParams, estimate_flow
from evflow.models.eraft_runner import WarmStartRunner

METRIC_KEYS = ("epe", "outlier_pct", "npe_1", "npe_2", "npe_3", "ae_deg")


def _to_torch(ev: dict, device) -> tuple[torch.Tensor, ...]:
    def conv(a):
        a = np.asarray(a)
        return torch.from_numpy(a.astype(np.float32 if np.issubdtype(a.dtype, np.floating)
                                         else np.int64)).to(device)
    return conv(ev["x"]), conv(ev["y"]), conv(ev["t"]), conv(ev["p"])


class ZeroPredictor:
    name = "zero"

    def __init__(self, height: int, width: int) -> None:
        self.shape = (2, height, width)

    def __call__(self, w: Window, ev: dict) -> np.ndarray:
        return np.zeros(self.shape)


class CMPredictor:
    name = "cm"

    def __init__(self, params: CMParams, height: int, width: int, device) -> None:
        self.params, self.h, self.w, self.device = params, height, width, device

    def __call__(self, w: Window, ev: dict) -> np.ndarray:
        x, y, t, p = _to_torch(ev, "cpu")
        torch.manual_seed(0)
        flow = estimate_flow(x, y, t, p, t_range=(w.t_start_us, w.t_end_us), height=self.h,
                             width=self.w, params=self.params, device=self.device)
        return flow.double().cpu().numpy()


def eraft_voxel(ev: dict | None, *, bins: int, height: int, width: int, device) -> torch.Tensor:
    """15-bin voxel grid as E-RAFT builds it: time normalised over the slice's own first..last
    event (deltaT = 1 µs if they coincide), signed polarity, mean/std over non-zero cells."""
    if ev is None or len(ev["t"]) == 0:
        return torch.zeros((bins, height, width), dtype=torch.float32, device=device)
    x, y, t, p = _to_torch(ev, device)
    t0, t1 = int(t[0]), int(t[-1])
    return voxel_grid(x, y, t, p, bins=bins, height=height, width=width,
                      t_range=(t0, t1 if t1 > t0 else t0 + 1), normalize=True)


class ERAFTPredictor:
    """R1. ``crop`` = (r0, r1, c0, c1) applied to the full-frame voxels (MVSEC: E-RAFT's
    256x256 centre crop) and the flow pasted back into a zero full frame. ``new_span_us``: if
    set (DSEC), R1's volumes are [s - span, s) and [s, s + span) for a window starting at s,
    E-RAFT's DSEC rule; otherwise [prev_start, s) and [s, e)."""

    name = "eraft"

    def __init__(self, runner: WarmStartRunner, slicer, *, bins: int, height: int, width: int,
                 device, crop: tuple[int, int, int, int] | None = None,
                 new_span_us: int | None = None) -> None:
        self.runner, self.slicer = runner, slicer
        self.bins, self.h, self.w, self.device = bins, height, width, device
        self.crop, self.new_span_us = crop, new_span_us
        self._state: dict[int, torch.Tensor | None] = {}   # window end µs -> flow_init
        self._vox: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    def _volume(self, a: int, b: int) -> torch.Tensor:
        key = (a, b)
        if key not in self._vox:
            v = eraft_voxel(self.slicer.get_events(a, b), bins=self.bins, height=self.h,
                            width=self.w, device=self.device)
            if self.crop is not None:
                r0, r1, c0, c1 = self.crop
                v = v[:, r0:r1, c0:c1]
            self._vox[key] = v
            while len(self._vox) > 16:
                self._vox.popitem(last=False)
        return self._vox[key]

    def __call__(self, w: Window, ev: dict) -> np.ndarray:
        s = w.t_start_us
        if self.new_span_us is not None:
            old, new = (s - self.new_span_us, s), (s, s + self.new_span_us)
        else:
            if w.prev_start_us is None:
                raise ValueError(f"window {w.index} has no predecessor for R1's previous volume")
            old, new = (w.prev_start_us, s), (s, w.t_end_us)
        self.runner.flow_init = self._state.pop(s, None) if w.prev_start_us is not None else None
        flow = self.runner(self._volume(*old), self._volume(*new)).double().cpu().numpy()
        self._state[w.t_end_us] = self.runner.flow_init
        if len(self._state) > 64:  # chains that were never continued
            self._state.pop(next(iter(self._state)))
        if self.crop is None:
            return flow
        r0, r1, c0, c1 = self.crop
        full = np.zeros((2, self.h, self.w))
        full[:, r0:r1, c0:c1] = flow
        return full


@dataclass
class DatasetAccess:
    """What a unit needs from a dataset: events, GT and the static masks."""

    slicer: object                                  # EventSlicer
    gt: Callable[[Window], tuple[np.ndarray, np.ndarray]]  # -> flow (2,H,W), valid (H,W)
    height: int
    width: int
    region: dict | None                             # MVSEC eval region; None = full frame
    sparse: bool                                    # AND events-in-window into the mask


def evaluate_windows(windows: list[Window], data: DatasetAccess, predictor,
                     *, log: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    """Per-window arrays: index, t_start_us, t_end_us, n_events, n_px, the metrics, fwl, and the
    prediction wall time. Windows with no valid pixel keep NaN metrics (skipped, counted)."""
    h, w = data.height, data.width
    crop = region_mask(data.region, h, w) if data.region is not None else None
    n = len(windows)
    out = {k: np.full(n, np.nan) for k in (*METRIC_KEYS, "fwl", "pred_s")}
    out.update(index=np.zeros(n, np.int64), t_start_us=np.zeros(n, np.int64),
               t_end_us=np.zeros(n, np.int64), n_events=np.zeros(n, np.int64),
               n_px=np.zeros(n, np.int64))
    t_log = time.time()
    for i, win in enumerate(windows):
        ev = data.slicer.get_events(win.t_start_us, win.t_end_us)
        if ev is None:
            raise ValueError(f"window {win.index} not covered by the event store")
        t0 = time.perf_counter()
        pred = predictor(win, ev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["pred_s"][i] = time.perf_counter() - t0
        gt_flow, valid = data.gt(win)
        eiw = events_in_window(ev["x"], ev["y"], h, w) if data.sparse else None
        m = flow_metrics(torch.from_numpy(pred), torch.from_numpy(np.asarray(gt_flow, np.float64)),
                         valid_gt=torch.from_numpy(valid),
                         events_in_window=None if eiw is None else torch.from_numpy(eiw),
                         crop=None if crop is None else torch.from_numpy(crop))
        for k in METRIC_KEYS:
            out[k][i] = m[k]
        out["n_px"][i] = m["n_px"]
        out["index"][i], out["t_start_us"][i], out["t_end_us"][i] = (win.index, win.t_start_us,
                                                                      win.t_end_us)
        out["n_events"][i] = len(ev["t"])
        out["fwl"][i] = window_fwl(ev, pred, win, h, w, crop)
        if log is not None and (time.time() - t_log > 30 or i == n - 1):
            log(f"{i + 1}/{n} windows, mean pred {np.nanmean(out['pred_s'][: i + 1]):.3f} s")
            t_log = time.time()
    return out


def window_fwl(ev: dict, flow: np.ndarray, win: Window, h: int, w: int,
               crop: np.ndarray | None) -> float:
    """FWL of the window's events (those whose pixel lies in ``crop``, if given); NaN when the
    zero-flow IWE has no variance (no events)."""
    x, y, t = np.asarray(ev["x"]), np.asarray(ev["y"]), np.asarray(ev["t"])
    if crop is not None:
        xi = np.clip(np.round(x).astype(np.intp), 0, w - 1)
        yi = np.clip(np.round(y).astype(np.intp), 0, h - 1)
        keep = crop[yi, xi]
        x, y, t = x[keep], y[keep], t[keep]
    if t.size == 0:
        return float("nan")
    xt = torch.from_numpy(x.astype(np.float64) if np.issubdtype(x.dtype, np.floating)
                          else x.astype(np.int64))
    yt = torch.from_numpy(y.astype(np.float64) if np.issubdtype(y.dtype, np.floating)
                          else y.astype(np.int64))
    val = fwl(xt, yt, torch.from_numpy(t.astype(np.int64)), torch.from_numpy(flow),
              t_range=(win.t_start_us, win.t_end_us), t_ref=1.0, height=h, width=w,
              interpolation="round")
    return val if np.isfinite(val) else float("nan")
