"""Summary of per-window arrays: per-window mean then mean over windows (configs/eval_mvsec.yaml
``averaging``), with moving-block bootstrap 95 % CIs (configs/evflow.yaml ``metrics.bootstrap``).
Windows with no valid pixel are skipped and counted; FWL is averaged over windows with events."""

from __future__ import annotations

import numpy as np

from evflow.eval.bootstrap import block_length, moving_block_ci
from evflow.eval.run import METRIC_KEYS


def summarise(per_window: dict[str, np.ndarray], *, block_seconds: float, resamples: int,
              ci: float, seed: int) -> dict:
    n = int(per_window["epe"].size)
    scored = per_window["n_px"] > 0
    block = block_length(per_window["t_start_us"], block_seconds)
    metrics, ci95 = {}, {}
    for k in (*METRIC_KEYS, "fwl"):
        v = per_window[k]
        ok = np.isfinite(v) & (scored if k != "fwl" else True)
        if not ok.any():
            metrics[k], ci95[k] = None, None
            continue
        metrics[k] = float(v[ok].mean())
        ci95[k] = list(moving_block_ci(v[ok], block=block, resamples=resamples, ci=ci, seed=seed))
    return {
        "metrics": metrics,
        "ci95": ci95,
        "n_windows": int(scored.sum()),
        "n_windows_total": n,
        "n_windows_skipped_no_valid_px": int(n - scored.sum()),
        "bootstrap": {"method": "moving_block", "block_windows": block,
                      "block_seconds": block_seconds, "resamples": resamples, "ci": ci,
                      "seed": seed},
        "mean_valid_px_per_window": float(per_window["n_px"][scored].mean()) if scored.any()
        else 0.0,
        "mean_events_per_window": float(per_window["n_events"].mean()) if n else 0.0,
        "mean_pred_s_per_window": float(np.nanmean(per_window["pred_s"])) if n else None,
    }
