"""Simulator-fidelity statistics and count-image alignment, real vs v2e (Phase 3 steps 2–3).

Both use the units of the corpus plan (``sim/corpus.py``) and treat real and synthetic events
identically (D-033):

- windows: the unit's owned APS dt=1 windows from ``stats_from_window`` on (warm-up excluded);
- events: both streams restricted to the owned span ``[aps[k0], aps[k1 + 1])`` before any
  statistic, including the per-pixel inter-event interval; blocks of ``block_us`` tile the span;
- §1.6 statistics from ``sim/stats.py`` (fixed histogram edges, so every W1 is recomputable from
  the saved histograms); pooled over units by adding histograms.

Alignment: per-window NCC of blurred count images (``stats.count_ncc``), with three negative
controls: the synthetic window ``s`` frames later (s = 1, 5: timing error) and the synthetic
stream mirrored in x (geometry error). The threshold is the Phase 2 tune value
(``configs/v2e_profiles.yaml: alignment_ncc_threshold``, D-029); nothing here sets or changes it.
Test-sequence statistics are reported only, never used for a choice (plan Phase 3 step 2).
"""

from __future__ import annotations

import numpy as np

from evflow.sim import stats as S

SHIFT_CONTROLS = (1, 5)


def owned_window_starts(unit: dict, dt: int = 1) -> np.ndarray:
    """Window starts k of a unit used for statistics: owned, after warm-up, k + dt in the unit."""
    k0 = max(int(unit["owned_window_starts"][0]), int(unit["stats_from_window"]))
    k1 = min(int(unit["owned_window_starts"][1]), int(unit["aps"][1]) - dt)
    return np.arange(k0, k1 + 1, dtype=np.int64)


def windows(aps_us: np.ndarray, starts: np.ndarray, dt: int = 1) -> np.ndarray:
    return np.stack([aps_us[starts], aps_us[starts + dt]], 1).astype(np.int64)


def blocks(span: tuple[int, int], block_us: int) -> np.ndarray:
    """Fixed blocks tiling ``span`` (last partial block dropped)."""
    starts = np.arange(span[0], span[1] - block_us + 1, block_us, dtype=np.int64)
    return np.stack([starts, starts + block_us], 1)


def restrict(e: dict, t0: int, t1: int) -> dict:
    a, b = np.searchsorted(e["t"], [t0, t1], side="left")
    return {k: e[k][a:b] for k in ("t", "x", "y", "p")}


def unit_stats(real: dict, syn: dict, aps_us: np.ndarray, unit: dict, block_us: int,
               width: int, height: int) -> dict:
    """§1.6 statistics of one unit for both streams over identical windows/blocks/span."""
    k = owned_window_starts(unit)
    wins = windows(aps_us, k)
    span = (int(wins[0, 0]), int(wins[-1, 1]))
    blk = blocks(span, block_us)
    out = {"n_windows": len(wins), "n_blocks": len(blk), "span_us": list(span)}
    for name, e in (("real", real), ("v2e", syn)):
        out[name] = S.slice_stats(restrict(e, *span), wins, blk, width, height)
    return out


def pool(units: list[dict]) -> dict:
    """Add histograms, counts and sample-weighted means over unit_stats results."""
    out = {"n_windows": sum(u["n_windows"] for u in units),
           "n_blocks": sum(u["n_blocks"] for u in units)}
    for name in ("real", "v2e"):
        hist = {k: np.sum([u[name]["hist"][k] for u in units], 0).astype(np.int64)
                for k in S.STATS}
        n = {k: int(sum(u[name]["n_samples"][k] for u in units)) for k in S.STATS}
        mean = {}
        for k in S.STATS:
            num = sum(u[name]["mean"][k] * u[name]["n_samples"][k] for u in units
                      if u[name]["mean"][k] is not None)
            mean[k] = float(num / n[k]) if n[k] else None
        span_s = sum((u["span_us"][1] - u["span_us"][0]) / 1e6 for u in units)
        n_ev = int(sum(u[name]["n_events"] for u in units))
        out[name] = {"hist": hist, "n_samples": n, "mean": mean, "n_events": n_ev,
                     "events_per_s": n_ev / span_s if span_s else None}
    out["span_s"] = round(sum((u["span_us"][1] - u["span_us"][0]) / 1e6 for u in units), 4)
    return out


def distances(pooled: dict) -> dict:
    """W1 per statistic (raw, and normalised by the real std as in calibration, D-027)."""
    raw, norm = {}, {}
    for k in S.STATS:
        hr, hs = pooled["real"]["hist"][k], pooled["v2e"]["hist"][k]
        e = S.EDGES[k]
        raw[k] = S.w1_hist(hr, hs, e)
        norm[k] = raw[k] / max(S.hist_std(hr, e), float(e[1] - e[0]))
    return {"w1": raw, "w1_normalised": norm,
            "mean_w1_normalised": float(np.mean(list(norm.values())))}


def unit_ncc(real: dict, syn: dict, aps_us: np.ndarray, unit: dict, width: int, height: int,
             blur_sigma: float, shifts=SHIFT_CONTROLS) -> dict[str, np.ndarray]:
    """Per-window NCC and its negative controls for one unit."""
    k = owned_window_starts(unit)
    wins = windows(aps_us, k)
    out = {"aligned": S.count_ncc(real, syn, wins, width, height, blur_sigma)}
    for s in shifts:
        ok = k + s + 1 <= int(unit["aps"][1])
        out[f"shift{s}"] = S.count_ncc(real, syn, wins[ok], width, height, blur_sigma,
                                       syn_wins=windows(aps_us, k[ok] + s))
    mir = {**syn, "x": (width - 1 - syn["x"].astype(np.int64)).astype(syn["x"].dtype)}
    out["mirror"] = S.count_ncc(real, mir, wins, width, height, blur_sigma)
    return out


def ncc_summary(values: dict[str, list[np.ndarray]], threshold: float) -> dict:
    """Median / p10 / p90 of the aligned NCC and the control medians, pooled over units."""
    v = np.concatenate(values["aligned"])
    out = {"n_windows": int(np.isfinite(v).sum()), "n_nan": int((~np.isfinite(v)).sum()),
           "median": float(np.nanmedian(v)), "p10": float(np.nanpercentile(v, 10)),
           "p90": float(np.nanpercentile(v, 90)), "threshold": threshold,
           "pass": bool(np.nanmedian(v) > threshold), "controls": {}}
    for name, arrs in values.items():
        if name == "aligned":
            continue
        c = np.concatenate(arrs)
        med = float(np.nanmedian(c))
        out["controls"][name] = {"median": med, "below_threshold": bool(med < threshold),
                                 "aligned_minus_control": out["median"] - med,
                                 "n_windows": int(np.isfinite(c).sum())}
    return out
