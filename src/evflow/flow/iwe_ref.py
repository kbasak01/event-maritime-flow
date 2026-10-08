"""Pure-numpy image of warped events (IWE), FWL and RSAT (IMPLEMENTATION_PLAN §1.6).

Conventions:
- Time is normalised over the window: ``tau = (t - t_start) / (t_end - t_start)`` in [0, 1).
- ``flow`` is a dense (2, H, W) forward flow in pixels over the window (channel 0 = u, 1 = v).
  Each event takes the flow at its own pixel (float coordinates: nearest pixel, half to even).
- Warp to the reference time ``t_ref`` (normalised): ``x' = x + (t_ref - tau) u``,
  ``y' = y + (t_ref - tau) v``.
- ``interpolation='round'``: each warped event adds 1 at the nearest pixel (half to even);
  ``'bilinear'``: weights ``max(0, 1 - |x' - xi|) * max(0, 1 - |y' - yi|)`` on the 4 neighbours.
  Contributions that land outside the image are dropped. Both match tudelft/event_flow
  ``utils/iwe.py`` (verified by ``tests/test_oracle_parity.py``).

Metrics:
- FWL (Stoffregen et al., ECCV 2020): ``Var(IWE(flow)) / Var(IWE(0))``, spatial variance over all
  pixels, all polarities pooled. > 1 is better; zero flow gives exactly 1.
- RSAT (Hagenaars et al., NeurIPS 2021): per pixel and polarity, average normalised timestamp of
  the warped events; sum of squares over pixels and both polarities divided by the number of
  pixels hit by any warped event; ratio flow / zero flow. < 1 is better; zero flow gives exactly 1.
  The 1e-9 guard in the per-pixel average follows event_flow ``loss/flow.py``.
"""

from __future__ import annotations

import numpy as np

INTERPOLATIONS = ("round", "bilinear")


def normalize_time(t, t_range: tuple[int, int]) -> np.ndarray:
    t0, t1 = t_range
    if not t1 > t0:
        raise ValueError(f"t_range must satisfy t_end > t_start, got {t_range}")
    return (np.asarray(t, np.int64) - int(t0)).astype(np.float64) / float(t1 - t0)


def _pixel_index(x, y, height: int, width: int):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(y.dtype, np.floating):
        x = np.round(x.astype(np.float64))
        y = np.round(y.astype(np.float64))
    xi = np.clip(x.astype(np.int64), 0, width - 1)
    yi = np.clip(y.astype(np.int64), 0, height - 1)
    return xi, yi


def sample_flow(flow, x, y) -> np.ndarray:
    """Per-event flow (N, 2) = flow[:, y, x] at each event's own pixel."""
    flow = np.asarray(flow, np.float64)
    xi, yi = _pixel_index(x, y, flow.shape[1], flow.shape[2])
    return np.stack([flow[0, yi, xi], flow[1, yi, xi]], axis=1)


def warp(x, y, tau, flow_ev, *, t_ref: float):
    dt = float(t_ref) - np.asarray(tau, np.float64)
    return (np.asarray(x, np.float64) + dt * flow_ev[:, 0],
            np.asarray(y, np.float64) + dt * flow_ev[:, 1])


def splat(xw, yw, *, height: int, width: int, interpolation: str, weights=None) -> np.ndarray:
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
    wts = np.ones(len(xw)) if weights is None else np.asarray(weights, np.float64)
    img = np.zeros(height * width, dtype=np.float64)
    if interpolation == "round":
        taps = [(np.round(xw), np.round(yw), wts)]
    else:
        x0, y0 = np.floor(xw), np.floor(yw)
        taps = []
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                w = (np.maximum(0.0, 1.0 - np.abs(xw - xi))
                     * np.maximum(0.0, 1.0 - np.abs(yw - yi)))
                taps.append((xi, yi, wts * w))
    for xi, yi, w in taps:
        ok = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        np.add.at(img, yi[ok].astype(np.int64) * width + xi[ok].astype(np.int64), w[ok])
    return img.reshape(height, width)


def iwe(
    x, y, t, flow, *, t_range, t_ref: float, height: int, width: int, interpolation: str,
    weights=None,
) -> np.ndarray:
    """Image of events warped to ``t_ref`` by ``flow`` (H, W)."""
    tau = normalize_time(t, t_range)
    xw, yw = warp(x, y, tau, sample_flow(flow, x, y), t_ref=t_ref)
    return splat(xw, yw, height=height, width=width, interpolation=interpolation, weights=weights)


def fwl(x, y, t, flow, *, t_range, t_ref: float, height: int, width: int,
        interpolation: str) -> float:
    a = dict(t_range=t_range, t_ref=t_ref, height=height, width=width,
             interpolation=interpolation)
    img = iwe(x, y, t, flow, **a)
    img0 = iwe(x, y, t, np.zeros_like(np.asarray(flow, np.float64)), **a)
    return float(img.var() / img0.var())


def _rsat_term(x, y, tau, p, flow, a) -> float:
    xw, yw = warp(x, y, tau, sample_flow(flow, x, y), t_ref=a["t_ref"])
    s = dict(height=a["height"], width=a["width"], interpolation=a["interpolation"])
    total, hit = 0.0, np.zeros((a["height"], a["width"]), dtype=bool)
    for pol in (1, 0):
        m = np.asarray(p) == pol
        cnt = splat(xw[m], yw[m], **s)
        ts = splat(xw[m], yw[m], weights=tau[m], **s) / (cnt + 1e-9)
        total += float(np.sum(ts**2))
        hit |= cnt > 0
    return total / float(hit.sum())


def rsat(x, y, t, p, flow, *, t_range, t_ref: float, height: int, width: int,
         interpolation: str) -> float:
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
    a = dict(t_ref=t_ref, height=height, width=width, interpolation=interpolation)
    tau = normalize_time(t, t_range)
    flow = np.asarray(flow, np.float64)
    return _rsat_term(x, y, tau, p, flow, a) / _rsat_term(x, y, tau, p, np.zeros_like(flow), a)
