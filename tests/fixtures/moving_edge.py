"""Analytic moving-edge event generator (Phase 1 fixture).

A set of straight step edges translates rigidly with velocity w = (u, v) px/s. Edge k is the line
n_k . X = c_k + (n_k . w) t, with unit normal n_k. A pixel centre X_p fires exactly once per edge,
at the time the edge crosses it:

    t = (n_k . X_p - c_k) / (n_k . w)

Polarity is ON when the bright side of the edge moves onto the pixel. Warping every event to a
reference time with the true flow puts all events of edge k back onto that edge's line, so the
correct flow gives a sharp image of warped events and the opposite flow a smeared one.

Everything is closed-form: no noise, no randomness beyond the seeded edge layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovingEdgeEvents:
    x: np.ndarray  # uint16
    y: np.ndarray  # uint16
    t: np.ndarray  # int64 µs, from t_start, non-decreasing
    p: np.ndarray  # int8 {0, 1}
    width: int
    height: int
    t_start_us: int
    t_end_us: int
    flow_px: np.ndarray  # (2, H, W) true forward flow over [t_start, t_end), pixels

    @property
    def velocity_px_per_s(self) -> tuple[float, float]:
        dt_s = (self.t_end_us - self.t_start_us) * 1e-6
        return float(self.flow_px[0, 0, 0] / dt_s), float(self.flow_px[1, 0, 0] / dt_s)


def moving_edges(
    *,
    width: int = 64,
    height: int = 48,
    velocity_px_per_s: tuple[float, float] = (400.0, -250.0),
    duration_us: int = 20_000,
    n_edges: int = 8,
    seed: int = 0,
    t_start_us: int = 0,
) -> MovingEdgeEvents:
    """Events from ``n_edges`` straight edges translating at ``velocity_px_per_s``.

    Edge orientations are spread over [0, pi) so both flow components are observable; offsets are
    chosen so every edge sweeps through the image during the window.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(velocity_px_per_s, dtype=np.float64)
    duration_s = duration_us * 1e-6

    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.ravel().astype(np.float64)
    ys = ys.ravel().astype(np.float64)

    all_x, all_y, all_t, all_p = [], [], [], []
    angles = (np.arange(n_edges) + rng.uniform(0.1, 0.9, n_edges)) * np.pi / n_edges
    for theta in angles:
        n = np.array([np.cos(theta), np.sin(theta)])
        speed_n = float(n @ w)
        if abs(speed_n) < 1e-9:
            continue
        proj = n[0] * xs + n[1] * ys
        # Edge position at t=0 chosen inside the projected image extent, so it crosses pixels.
        c = rng.uniform(proj.min(), proj.max())
        t_cross_s = (proj - c) / speed_n
        inside = (t_cross_s >= 0.0) & (t_cross_s < duration_s)
        contrast = 1 if rng.random() < 0.5 else -1  # +1: bright on the +n side
        # The edge moves along sign(speed_n) * n; the side behind it covers the pixel.
        # Behind = -sign(speed_n) * n side; bright there iff contrast * -sign(speed_n) > 0.
        on = contrast * -np.sign(speed_n) > 0
        t_us = np.floor(t_cross_s[inside] * 1e6).astype(np.int64) + t_start_us
        all_x.append(xs[inside].astype(np.uint16))
        all_y.append(ys[inside].astype(np.uint16))
        all_t.append(t_us)
        all_p.append(np.full(t_us.shape, 1 if on else 0, dtype=np.int8))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    p = np.concatenate(all_p)
    order = np.argsort(t, kind="stable")
    flow = np.empty((2, height, width), dtype=np.float64)
    flow[0] = w[0] * duration_s
    flow[1] = w[1] * duration_s
    return MovingEdgeEvents(
        x=x[order], y=y[order], t=t[order], p=p[order], width=width, height=height,
        t_start_us=t_start_us, t_end_us=t_start_us + duration_us, flow_px=flow,
    )
