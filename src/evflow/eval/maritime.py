"""T3 evaluation on the maritime corpus (IMPLEMENTATION_PLAN §1.6, Phase 7 step 4; D-057).

Units (all from the calibrated ×1.0 maritime corpus, D-054; 640×360; windows of one source-frame
interval, window k = frames k → k+1, first 2 windows of each v2e unit dropped as in training):

| dataset | sequence      | windows                                 | slices                    |
|---------|---------------|-----------------------------------------|---------------------------|
| smd     | <video stem>  | every window of the smd_test segment    | all                       |
| lars    | lars_eval_kf  | the keyframe window (k−1 → keyframe k)  | all, water, sky, obstacle |
| lars    | lars_eval_all | every window of the 40 lars_eval clips  | all                       |

Per window and slice (``all`` = the full frame; a class slice = the keyframe's LaRS label resized
to 640×360 with nearest neighbour, ignore pixels in no class):

- **T3a** (RAFT teacher, D-055; never "EPE"): ``teacher_epe``, ``teacher_npe_{1,3}``,
  ``teacher_ae_deg`` = ``flow.metrics.flow_metrics`` with the teacher flow as reference and the
  mask forward–backward-consistent ∧ events-in-window ∧ slice; ``teacher_mask_coverage`` =
  consistent px / slice px; ``events_px_fraction`` = px with events / slice px.
- **T3b** (self-consistency only): ``fwl`` and ``rsat`` (round, t_ref = 1) of the window's
  events whose pixel lies in the slice.

Summaries: mean over windows with a value; 95 % CIs by moving-block bootstrap over the windows of
one SMD video (1 s blocks) and by a cluster bootstrap over clips for LaRS (windows of one clip
resampled together).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from evflow.eval.bootstrap import block_length, moving_block_ci
from evflow.eval.run import window_fwl
from evflow.eval.windows import Window, events_in_window
from evflow.flow.fwl import rsat
from evflow.flow.metrics import flow_metrics

WARMUP = 2
SLICES = ("all", "water", "sky", "obstacle")
T3A_KEYS = ("teacher_epe", "teacher_npe_1", "teacher_npe_3", "teacher_ae_deg",
            "teacher_mask_coverage", "events_px_fraction")
T3B_KEYS = ("fwl", "rsat")
SEQUENCES = {"lars": ("lars_eval_kf", "lars_eval_all")}
LARS_CORPUS = "lars_eval_ct100"
SMD_CORPUS = "smd_test"


class ArraySlicer:
    """``EventSlicer``-like access to one unit's events held in RAM (``train.data.EventArrays``,
    which clamps a ≤ 1 ms end shortfall of a v2e store, D-048)."""

    def __init__(self, arrays) -> None:
        self.a = arrays

    def get_events(self, t0: int, t1: int) -> dict:
        return self.a.window(int(t0), int(t1))


@dataclass
class Segment:
    """The windows of one v2e unit, with its teacher store and per-window class masks."""

    key: str
    slicer: ArraySlicer
    windows: list[Window]
    local_k: list[int]                      # teacher window index of each window
    teacher: Path
    masks: list[dict[str, np.ndarray] | None] = field(default_factory=list)
    store_sha256: str = ""


def unit_windows(n_frames: int, frame_t: np.ndarray, ks) -> tuple[list[Window], list[int]]:
    """Windows k → k+1 (absolute µs); the predecessor is window k−1 (ends where k starts)."""
    out = []
    for k in ks:
        k = int(k)
        if not 0 <= k <= n_frames - 2:
            raise ValueError(f"window {k} outside {n_frames} frames")
        out.append(Window(k, int(frame_t[k]), int(frame_t[k + 1]),
                          int(frame_t[k - 1]) if k >= 1 else None))
    return out, [int(k) for k in ks]


def segments(paths: dict, dataset: str, sequence: str, *, size=(640, 360)) -> list[Segment]:
    """The evaluation segments of a T3 unit (reads the maritime corpus plan and stores)."""
    from evflow.io import lars as lars_io
    from evflow.io.frames import frame_times_us
    from evflow.sim import maritime_corpus as mc
    from evflow.train.data import EventArrays

    plan = mc.load_plan(paths)
    root = mc.out_root(paths)
    troot = Path(paths["outputs"]["events"]) / "teacher"
    if dataset == "smd":
        units = [u for u in plan["units"] if u["corpus"] == SMD_CORPUS
                 and Path(u["video"]).stem == sequence]
    elif dataset == "lars" and sequence in SEQUENCES["lars"]:
        units = [u for u in plan["units"] if u["corpus"] == LARS_CORPUS]
    else:
        raise ValueError(f"unknown T3 unit {dataset}/{sequence}")
    if not units:
        raise ValueError(f"no corpus units for {dataset}/{sequence}")
    out = []
    for u in units:
        if u["split"] not in ("smd_test", "lars_eval"):
            raise ValueError(f"{u['key']}: T3 evaluation reads smd_test / lars_eval only")
        man = json.loads((root / u["corpus"] / f"{u['name']}.json").read_text())
        n = int(u["n_frames"])
        ft = frame_times_us(n, float(man["input_frame_rate"]),
                            int(man["conversion"]["t_offset_us"]))
        if sequence == "lars_eval_kf":
            ks = [j - 1 for j in u["keyframes_local"]]
            if min(ks) < WARMUP:
                raise ValueError(f"{u['key']}: keyframe window inside the warm-up")
        else:
            ks = list(range(WARMUP, n - 1))
        wins, local = unit_windows(n, ft, ks)
        lo = min(w.t_start_us for w in wins)
        lo = min([lo] + [w.prev_start_us for w in wins if w.prev_start_us is not None])
        arr = EventArrays.from_store(root / u["corpus"] / f"{u['name']}.h5", lo,
                                     max(w.t_end_us for w in wins))
        masks: list = [None] * len(wins)
        if sequence == "lars_eval_kf":
            for i, kf in enumerate(u["keyframes"]):
                masks[i] = lars_io.class_masks(lars_io.semantic_mask(
                    paths["lars"]["root"], u["subset"], kf, size=size))
        tp = troot / u["corpus"] / f"{u['name']}.h5"
        out.append(Segment(u["key"], ArraySlicer(arr), wins, local, tp, masks,
                           man["store_sha256"]))
    return out


def window_rsat(ev: dict, flow: np.ndarray, win: Window, h: int, w: int,
                keep_px: np.ndarray | None) -> float:
    """RSAT (round, t_ref = 1) of the window's events whose pixel lies in ``keep_px``."""
    x, y = np.asarray(ev["x"]).astype(np.int64), np.asarray(ev["y"]).astype(np.int64)
    t, p = np.asarray(ev["t"], np.int64), np.asarray(ev["p"]).astype(np.int64)
    if keep_px is not None:
        k = keep_px[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
        x, y, t, p = x[k], y[k], t[k], p[k]
    if t.size < 2:
        return float("nan")
    with np.errstate(all="ignore"):
        try:
            v = rsat(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(t),
                     torch.from_numpy(p), torch.from_numpy(flow),
                     t_range=(win.t_start_us, win.t_end_us), t_ref=1.0, height=h, width=w,
                     interpolation="round")
        except ZeroDivisionError:
            return float("nan")
    return v if np.isfinite(v) else float("nan")


def evaluate(segs: list[Segment], predictor_for: Callable[[Segment], Callable], *,
             height: int, width: int, slices=SLICES,
             log: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    """Per-window arrays ``<metric>__<slice>`` plus index, group (segment), t_start_us,
    t_end_us, n_events, pred_s. Class slices exist only where the window has masks."""
    from evflow.teacher.cache import TeacherCache

    n = sum(len(s.windows) for s in segs)
    keys = [f"{k}__{s}" for s in slices for k in (*T3A_KEYS, *T3B_KEYS, "n_px", "slice_px")]
    out = {k: np.full(n, np.nan) for k in (*keys, "pred_s")}
    out.update(index=np.zeros(n, np.int64), group=np.zeros(n, np.int64),
               t_start_us=np.zeros(n, np.int64), t_end_us=np.zeros(n, np.int64),
               n_events=np.zeros(n, np.int64))
    i, t_log = 0, time.time()
    for g, seg in enumerate(segs):
        pred_fn = predictor_for(seg)
        with TeacherCache(seg.teacher) as tc:
            for win, k, masks in zip(seg.windows, seg.local_k, seg.masks or
                                     [None] * len(seg.windows), strict=True):
                ev = seg.slicer.get_events(win.t_start_us, win.t_end_us)
                t0 = time.perf_counter()
                pred = np.asarray(pred_fn(win, ev), np.float64)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                out["pred_s"][i] = time.perf_counter() - t0
                tf, tv = tc.get(k)
                tv = np.asarray(tv, bool)
                eiw = events_in_window(np.asarray(ev["x"]).astype(np.int64),
                                       np.asarray(ev["y"]).astype(np.int64), height, width)
                for s in slices:
                    if s == "all":
                        cm = np.ones((height, width), bool)
                    elif masks is None:
                        continue
                    else:
                        cm = masks[s]
                    npx = int(cm.sum())
                    out[f"slice_px__{s}"][i] = npx
                    if npx == 0:
                        continue
                    m = flow_metrics(torch.from_numpy(pred), torch.from_numpy(
                        np.asarray(tf, np.float64)), valid_gt=torch.from_numpy(tv),
                        events_in_window=torch.from_numpy(eiw),
                        class_mask=torch.from_numpy(cm))
                    out[f"n_px__{s}"][i] = m["n_px"]
                    if m["n_px"] > 0:
                        out[f"teacher_epe__{s}"][i] = m["epe"]
                        out[f"teacher_npe_1__{s}"][i] = m["npe_1"]
                        out[f"teacher_npe_3__{s}"][i] = m["npe_3"]
                        out[f"teacher_ae_deg__{s}"][i] = m["ae_deg"]
                    out[f"teacher_mask_coverage__{s}"][i] = float((tv & cm).sum()) / npx
                    out[f"events_px_fraction__{s}"][i] = float((eiw & cm).sum()) / npx
                    out[f"fwl__{s}"][i] = window_fwl(ev, pred, win, height, width,
                                                     None if s == "all" else cm)
                    out[f"rsat__{s}"][i] = window_rsat(ev, pred, win, height, width,
                                                       None if s == "all" else cm)
                out["index"][i], out["group"][i] = k, g
                out["t_start_us"][i], out["t_end_us"][i] = win.t_start_us, win.t_end_us
                out["n_events"][i] = len(ev["t"])
                i += 1
                if log is not None and (time.time() - t_log > 30 or i == n):
                    log(f"{i}/{n} windows, mean pred {np.nanmean(out['pred_s'][:i]):.3f} s")
                    t_log = time.time()
    return out


# ---------------------------------------------------------------------------------------------
# summaries

def cluster_ci(values: np.ndarray, groups: np.ndarray, *, resamples: int = 1000,
               ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the window mean when whole groups (clips) are resampled with
    replacement (window-weighted: the mean over all windows of the drawn groups)."""
    v = np.asarray(values, np.float64)
    g = np.asarray(groups)
    if v.size == 0:
        raise ValueError("no values")
    ids, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=ids.size)
    cnts = np.bincount(inv, minlength=ids.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, ids.size, size=(resamples, ids.size))
    stats = sums[draw].sum(1) / cnts[draw].sum(1)
    a = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [a, 1.0 - a])
    return float(lo), float(hi)


def summarise_slice(pw: dict, s: str, keys, *, cluster: bool, block_seconds: float,
                    resamples: int, ci: float, seed: int) -> dict:
    metrics, ci95, n_win = {}, {}, {}
    for k in keys:
        v = pw[f"{k}__{s}"]
        ok = np.isfinite(v)
        n_win[k] = int(ok.sum())
        if not ok.any():
            metrics[k], ci95[k] = None, None
            continue
        metrics[k] = float(v[ok].mean())
        if cluster:
            ci95[k] = list(cluster_ci(v[ok], pw["group"][ok], resamples=resamples, ci=ci,
                                      seed=seed))
        else:
            blk = block_length(pw["t_start_us"][ok], block_seconds)
            ci95[k] = list(moving_block_ci(v[ok], block=blk, resamples=resamples, ci=ci,
                                           seed=seed))
    sp = pw[f"slice_px__{s}"]
    has = np.isfinite(sp)
    return {"metrics": metrics, "ci95": ci95, "n_windows_per_metric": n_win,
            "n_windows": int(has.sum()),
            "mean_slice_px": float(sp[has].mean()) if has.any() else None,
            "mean_scored_px": float(np.nanmean(pw[f"n_px__{s}"][has])) if has.any() else None}
