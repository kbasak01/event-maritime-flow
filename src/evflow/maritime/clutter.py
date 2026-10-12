"""Sea-clutter event-budget study (IMPLEMENTATION_PLAN §0(5), Phase 7 step 5; D-058).

On the keyframe window (frame k−1 → keyframe k, one frame interval) of each of the 40
``lars_eval`` clips, converted by v2e at three contrast-threshold multipliers (0.67, 1.0, 1.5 ×
the calibrated profile, D-054), each event is assigned the LaRS class of its pixel in the
keyframe's semantic mask (resized to the 640×360 event grid, nearest). Per clip and class c:

- ``event_share``  = events in c / all events in the window (ignore pixels included in the total)
- ``pixel_share``  = pixels in c / all pixels
- ``enrichment``   = event_share / pixel_share (> 1: the class emits more than its area)
- ``events_per_px`` = events in c / pixels in c, per frame interval (LaRS frame rate assumed,
  D-024, so no per-second rate is formed)

Pooled values sum events and pixels over clips (event-weighted); CIs are cluster bootstraps over
clips. Scene attributes are the LaRS keyframe annotations (``io.lars.load_attributes``; only
fields that exist): per level n clips and the mean water values (levels with n < ``MIN_N`` are
listed with n only), and Spearman's ρ of water enrichment / water events-per-pixel against the
ordinal (waves, reflections) and binary attributes with a clip-bootstrap CI.

Everything here is **synthetic** (v2e events from LaRS frames, which inherit the frames' dynamic
range and blur; §0(1)): an event-budget characterisation of the simulator on maritime content,
not a measurement of a real sensor.
"""

from __future__ import annotations

import numpy as np

from evflow.io.lars import CLASSES, ORDINAL

MIN_N = 3
BINARY = ("special.glitter", "special.wakes", "special.extra_bright", "special.extra_dark",
          "special.fog", "special.dirty_lens", "special.plants_debris", "special.rain")
NOMINAL = ("scene_type", "lighting")


def window_class_counts(x: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Events and pixels per class (+ ``ignore``) of one window; integer event coordinates."""
    x = np.asarray(x).astype(np.intp)
    y = np.asarray(y).astype(np.intp)
    h, w = masks["water"].shape
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not ok.all():
        raise ValueError("event outside the label grid")
    out = {"n_events": int(x.size), "n_px": int(h * w), "classes": {}}
    for c in (*CLASSES, "ignore"):
        m = masks[c]
        out["classes"][c] = {"events": int(m[y, x].sum()), "px": int(m.sum())}
    return out


def clip_stats(counts: list[dict]) -> dict:
    """Sum the keyframe windows of one clip and derive the per-class quantities (pixels are
    summed over windows too, so ``events_per_px`` is per pixel per window = per frame
    interval)."""
    n_ev = sum(c["n_events"] for c in counts)
    n_px = sum(c["n_px"] for c in counts)
    out = {"n_events": n_ev, "n_px": n_px, "classes": {}}
    for c in CLASSES:
        ev = sum(k["classes"][c]["events"] for k in counts)
        px = sum(k["classes"][c]["px"] for k in counts)
        es = ev / n_ev if n_ev else float("nan")
        ps = px / n_px if n_px else float("nan")
        out["classes"][c] = {"events": ev, "px": px, "event_share": es, "pixel_share": ps,
                             "enrichment": es / ps if ps > 0 and n_ev else float("nan"),
                             "events_per_px": ev / px if px else float("nan")}
    return out


def pooled(clips: list[dict], *, resamples: int = 1000, ci: float = 0.95,
           seed: int = 0) -> dict:
    """Event-weighted pooled share, pixel share, enrichment and events/px per class with
    cluster-bootstrap CIs over clips."""
    rng = np.random.default_rng(seed)
    n = len(clips)
    draw = rng.integers(0, n, size=(resamples, n))
    tot_ev = np.array([c["n_events"] for c in clips], np.float64)
    tot_px = np.array([c["n_px"] for c in clips], np.float64)
    a = (1.0 - ci) / 2.0
    out = {}
    for cls in CLASSES:
        ev = np.array([c["classes"][cls]["events"] for c in clips], np.float64)
        px = np.array([c["classes"][cls]["px"] for c in clips], np.float64)

        def stats(idx, ev=ev, px=px):
            e, p = ev[idx].sum(-1), px[idx].sum(-1)
            te, tp = tot_ev[idx].sum(-1), tot_px[idx].sum(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                es, ps = e / te, p / tp
                return {"event_share": es, "pixel_share": ps, "enrichment": es / ps,
                        "events_per_px": e / p}
        point = stats(np.arange(n))
        boot = stats(draw)
        out[cls] = {}
        for k, v in point.items():
            b = boot[k][np.isfinite(boot[k])]
            out[cls][k] = {"mean": float(v),
                           "ci95": [float(np.quantile(b, a)), float(np.quantile(b, 1 - a))]
                           if b.size else None, "n_valid_resamples": int(b.size)}
        out[cls]["n_clips_with_class"] = int((px > 0).sum())
    return out


def spearman_ci(xs: np.ndarray, ys: np.ndarray, *, resamples: int = 1000, ci: float = 0.95,
                seed: int = 0) -> dict:
    """Spearman's ρ (average ranks for ties) with a percentile bootstrap CI over clips."""
    from scipy.stats import spearmanr

    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 2 * MIN_N or np.unique(xs).size < 2:
        return {"rho": None, "ci95": None, "n": int(xs.size)}
    rho = float(spearmanr(xs, ys).statistic)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(resamples):
        i = rng.integers(0, xs.size, xs.size)
        if np.unique(xs[i]).size > 1 and np.unique(ys[i]).size > 1:
            bs.append(spearmanr(xs[i], ys[i]).statistic)
    a = (1.0 - ci) / 2.0
    return {"rho": rho, "ci95": [float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))]
            if bs else None, "n": int(xs.size), "n_valid_resamples": len(bs),
            "resamples": resamples}


def attribute_breakdown(clips: list[dict], attrs: list[dict], *, seed: int = 0) -> dict:
    """Per attribute: levels with n and mean water values; ρ for ordinal and binary ones."""
    water_enr = np.array([c["classes"]["water"]["enrichment"] for c in clips])
    water_epp = np.array([c["classes"]["water"]["events_per_px"] for c in clips])
    water_es = np.array([c["classes"]["water"]["event_share"] for c in clips])
    out = {}
    for name in (*ORDINAL, *BINARY, *NOMINAL):
        vals = [a.get(name) for a in attrs]
        levels: dict = {}
        for i, v in enumerate(vals):
            levels.setdefault("null" if v is None else str(v).lower(), []).append(i)
        rows = {}
        for lv, idx in sorted(levels.items()):
            idx = np.array(idx)
            if idx.size >= MIN_N:
                rows[lv] = {"n": int(idx.size),
                            "water_event_share": float(np.nanmean(water_es[idx])),
                            "water_enrichment": float(np.nanmean(water_enr[idx])),
                            "water_events_per_px": float(np.nanmean(water_epp[idx]))}
            else:
                rows[lv] = {"n": int(idx.size), "reported": False,
                            "reason": f"fewer than {MIN_N} clips"}
        entry = {"levels": rows}
        if name in ORDINAL or name in BINARY:
            order = ORDINAL.get(name, ("false", "true"))
            rank = np.array([order.index(str(v).lower()) if v is not None and
                             str(v).lower() in order else np.nan for v in vals], np.float64)
            counts = [int((rank == r).sum()) for r in range(len(order))]
            if sum(c >= MIN_N for c in counts) >= 2:
                entry["spearman_water_enrichment"] = spearman_ci(rank, water_enr, seed=seed)
                entry["spearman_water_events_per_px"] = spearman_ci(rank, water_epp, seed=seed)
            else:
                entry["spearman"] = {"reported": False,
                                     "reason": f"fewer than two levels with >= {MIN_N} clips",
                                     "level_counts": dict(zip(order, counts, strict=True))}
        out[name] = entry
    return out
