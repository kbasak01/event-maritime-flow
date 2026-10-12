"""Tune-split evaluation for epoch selection (Phase 5 step 2): the MVSEC protocol masks of
configs/eval_mvsec.yaml on full frames.

Per window: ``flow.metrics.flow_metrics`` with valid GT AND events-in-window (of the evaluated
stream's own events, so a v2e arm's selection reads no real event) AND the ``center_256_no_hood``
region; FWL (``interpolation='round'``, t_ref = 1, CONVENTIONS) of the events inside the region.
Averaging: per-window mean, then mean over windows with >= 1 valid pixel (FWL: over windows with
events), as ``eval/summary.py``. Optional collapse diagnostics (``losses.cm.cm_diagnostics``) on
a fixed, evenly spaced subset of dt=1 windows.
"""

from __future__ import annotations

import numpy as np
import torch

from evflow.eval.run import METRIC_KEYS
from evflow.eval.windows import region_mask
from evflow.flow.metrics import ae_map, epe_map
from evflow.flow.metrics_ref import NPE_THRESHOLDS, OUTLIER_ABS_PX, OUTLIER_REL
from evflow.losses.cm import WindowEvents, cm_diagnostics
from evflow.models.evflownet import pad_to_multiple, unpad
from evflow.train.data import WindowDataset, batch_inputs, cat_events, collate


@torch.no_grad()
def predict(model, batch: dict, rep: str, height: int, width: int, device,
            amp_dtype: torch.dtype | None) -> torch.Tensor:
    inp = batch_inputs(batch, rep, height, width, device)
    inp, hw = pad_to_multiple(inp, 16)
    with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                        enabled=amp_dtype is not None):
        out = model(inp)[-1]
    return unpad(out.float(), hw)


def _subset(idx: np.ndarray, n: int) -> set[int]:
    if n <= 0 or idx.size == 0:
        return set()
    return {int(i) for i in idx[np.unique(np.linspace(0, idx.size - 1, min(n, idx.size))
                                          .round().astype(int))]}


@torch.no_grad()
def batch_window_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                         crop: torch.Tensor, ev: tuple[torch.Tensor, ...],
                         duration_us: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-window metrics of a batch, vectorised; equal to ``flow_metrics`` (valid AND
    events-in-window AND crop) and ``fwl`` (round, t_ref = 1, events inside ``crop``) window by
    window (tests/test_train_tune_eval.py). ``ev`` = (x, y, t, b) int64, t relative to the window
    start, b = sample index. Windows without a masked pixel / without events give NaN."""
    bsz, _, h, w = pred.shape
    x, y, t, b = ev
    pix = (b * h + y) * w + x
    eiw = torch.zeros(bsz * h * w, dtype=torch.bool, device=pred.device)
    eiw[pix] = True
    mask = valid.bool() & eiw.view(bsz, h, w) & crop[None]
    p64, g64 = pred.double(), gt.double()
    e = epe_map(p64, g64)
    n = mask.sum((1, 2)).double()
    nan = torch.full_like(n, float("nan"))

    def mean(v):
        return torch.where(n > 0, (v * mask).sum((1, 2)) / n.clamp(min=1), nan)

    gmag = torch.sqrt(g64[:, 0] ** 2 + g64[:, 1] ** 2)   # the metrics_ref formula (D-052)
    out = {"epe": mean(e),
           "outlier_pct": 100.0 * mean(((e > OUTLIER_ABS_PX) & (e > OUTLIER_REL * gmag)).double()),
           "ae_deg": mean(ae_map(p64, g64)), "n_px": n}
    for k in NPE_THRESHOLDS:
        out[f"npe_{k}"] = 100.0 * mean((e > k).double())
    # FWL and flow magnitude over the events inside the crop
    keep = crop[y, x]
    xk, yk, bk = x[keep], y[keep], b[keep]
    tau = t[keep].double() / duration_us.to(pred.device)[bk].double()
    u = p64[bk, 0, yk, xk]
    v = p64[bk, 1, yk, xk]
    xw = torch.round(xk.double() + (1.0 - tau) * u)
    yw = torch.round(yk.double() + (1.0 - tau) * v)
    inb = (xw >= 0) & (xw < w) & (yw >= 0) & (yw < h)
    img = torch.bincount(((bk[inb] * h + yw[inb].long()) * w + xw[inb].long()),
                         minlength=bsz * h * w).double().view(bsz, -1)
    img0 = torch.bincount((bk * h + yk) * w + xk, minlength=bsz * h * w).double().view(bsz, -1)
    var0 = img0.var(dim=1, unbiased=False)
    out["fwl"] = torch.where(var0 > 0, img.var(dim=1, unbiased=False) / var0, nan)
    cnt = torch.bincount(bk, minlength=bsz).double()
    mag = torch.zeros(bsz, dtype=torch.float64, device=pred.device).index_add_(
        0, bk, torch.sqrt(u * u + v * v))
    out["flow_mag"] = torch.where(cnt > 0, mag / cnt.clamp(min=1), nan)
    return out


@torch.no_grad()
def evaluate(model, ds: WindowDataset, *, rep: str, region: dict, device, batch_size: int = 16,
             amp_dtype: torch.dtype | None = None, diag_windows: int = 0,
             num_workers: int = 0) -> dict[int, dict]:
    """{dt: {"metrics": {...}, "n_windows": .., "diag": {...}}} over every window of ``ds``."""
    was_training = model.training
    model.eval()
    h, w = ds.h, ds.w
    crop = torch.from_numpy(region_mask(region, h, w)).to(device)
    dts = np.array([r.dt for r in ds.refs])
    diag_idx = _subset(np.flatnonzero(dts == min(ds.dt_sizes)), diag_windows)
    keys = (*METRIC_KEYS, "fwl", "flow_mag", "n_px")
    per: dict[int, dict] = {}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, collate_fn=collate)
    i0 = 0
    for batch in loader:
        pred = predict(model, batch, rep, h, w, device, amp_dtype)
        x, y, t, _, b = cat_events(batch, device)
        m = batch_window_metrics(pred, batch["flow"].to(device), batch["mask"].to(device), crop,
                                 (x, y, t, b), batch["duration_us"])
        m = {k: v.cpu().numpy() for k, v in m.items()}
        for j in range(pred.shape[0]):
            acc = per.setdefault(int(batch["dt"][j]), {k: [] for k in keys})
            for k in keys:
                acc[k].append(float(m[k][j]))
            if (i0 + j) in diag_idx:
                sel = b == j
                dur = int(batch["duration_us"][j])
                d = cm_diagnostics(pred[j], WindowEvents(x[sel], y[sel], t[sel].double() / dur))
                dacc = acc.setdefault("diag", {})
                for k, v in d.items():
                    dacc.setdefault(k, []).append(v)
        i0 += pred.shape[0]
    if was_training:
        model.train()
    out = {}
    for dt, acc in per.items():
        n_px = np.asarray(acc["n_px"])
        scored = n_px > 0
        metrics = {}
        for k in METRIC_KEYS:
            v = np.asarray(acc[k], float)[scored]
            metrics[k] = float(np.nanmean(v)) if v.size else None
        for k in ("fwl", "flow_mag"):
            v = np.asarray(acc[k], float)
            metrics[k] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        diag = {k: float(np.nanmean(v)) for k, v in acc.get("diag", {}).items()}
        out[dt] = {"metrics": metrics, "n_windows": int(scored.sum()),
                   "n_windows_total": int(n_px.size), "diag": diag}
    return out
