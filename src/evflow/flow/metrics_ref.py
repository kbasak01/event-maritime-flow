"""Pure-numpy reference flow metrics (IMPLEMENTATION_PLAN §1.6, docs/CONVENTIONS.md).

Flow arrays are (2, H, W), channel 0 = u (x, right), 1 = v (y, down), forward flow in pixels over
the window. Every metric is computed over an explicit boolean mask

    mask = valid_gt AND events_in_window AND crop AND class_mask

where ``valid_gt`` is required and the others are optional (``None`` = not applied). Nothing is
derived inside: in particular "GT == 0 means invalid" (EV-FlowNet's rule) is the caller's job when
building ``valid_gt``.

Definitions (per window; averaging over windows is the evaluator's job, see
``configs/eval_mvsec.yaml``):
- ``epe``: mean Euclidean endpoint error, px.
- ``outlier_pct``: % of pixels with EPE > 3 px AND EPE > 0.05 |GT| (MVSEC/KITTI convention, both
  strict).
- ``npe_N``: % of pixels with EPE > N px, N in {1, 2, 3} (DSEC convention, strict).
- ``ae_deg``: mean angle between (u, v, 1) and (u_gt, v_gt, 1), degrees.
An empty mask gives NaN for every metric and ``n_px = 0``.
"""

from __future__ import annotations

import numpy as np

OUTLIER_ABS_PX = 3.0
OUTLIER_REL = 0.05
NPE_THRESHOLDS = (1, 2, 3)


def crop_mask(
    height: int, width: int, *, rows: tuple[int, int], cols: tuple[int, int]
) -> np.ndarray:
    """Boolean (H, W) mask that keeps ``rows[0] <= r < rows[1]`` and ``cols[0] <= c < cols[1]``."""
    m = np.zeros((height, width), dtype=bool)
    m[rows[0] : rows[1], cols[0] : cols[1]] = True
    return m


def eval_mask(*, valid_gt, events_in_window=None, crop=None, class_mask=None) -> np.ndarray:
    mask = np.asarray(valid_gt, dtype=bool).copy()
    for m in (events_in_window, crop, class_mask):
        if m is not None:
            m = np.asarray(m, dtype=bool)
            if m.shape != mask.shape:
                raise ValueError(f"mask shape {m.shape} != valid_gt shape {mask.shape}")
            mask &= m
    return mask


def epe_map(flow_pred, flow_gt) -> np.ndarray:
    d = np.asarray(flow_pred, np.float64) - np.asarray(flow_gt, np.float64)
    return np.sqrt(d[0] ** 2 + d[1] ** 2)


def ae_map(flow_pred, flow_gt) -> np.ndarray:
    u, v = np.asarray(flow_pred, np.float64)
    ug, vg = np.asarray(flow_gt, np.float64)
    c = (1.0 + u * ug + v * vg) / np.sqrt((1.0 + u * u + v * v) * (1.0 + ug * ug + vg * vg))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def flow_metrics(
    flow_pred, flow_gt, *, valid_gt, events_in_window=None, crop=None, class_mask=None
) -> dict[str, float]:
    flow_pred = np.asarray(flow_pred, np.float64)
    flow_gt = np.asarray(flow_gt, np.float64)
    if flow_pred.shape != flow_gt.shape or flow_pred.ndim != 3 or flow_pred.shape[0] != 2:
        raise ValueError(f"flow shapes must both be (2, H, W): {flow_pred.shape}, {flow_gt.shape}")
    if np.asarray(valid_gt).shape != flow_gt.shape[1:]:
        raise ValueError(f"valid_gt shape {np.asarray(valid_gt).shape} != {flow_gt.shape[1:]}")
    mask = eval_mask(valid_gt=valid_gt, events_in_window=events_in_window, crop=crop,
                     class_mask=class_mask)
    n = int(mask.sum())
    keys = ["epe", "outlier_pct", *(f"npe_{k}" for k in NPE_THRESHOLDS), "ae_deg"]
    if n == 0:
        return {**{k: float("nan") for k in keys}, "n_px": 0}
    e = epe_map(flow_pred, flow_gt)[mask]
    gt_mag = np.sqrt(flow_gt[0] ** 2 + flow_gt[1] ** 2)[mask]
    out = {
        "epe": float(e.mean()),
        "outlier_pct": float(100.0 * np.mean((e > OUTLIER_ABS_PX) & (e > OUTLIER_REL * gt_mag))),
    }
    for k in NPE_THRESHOLDS:
        out[f"npe_{k}"] = float(100.0 * np.mean(e > k))
    out["ae_deg"] = float(ae_map(flow_pred, flow_gt)[mask].mean())
    out["n_px"] = n
    return out
