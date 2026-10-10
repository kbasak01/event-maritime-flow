"""M0b: per-window patch-wise contrast maximisation (classical baseline, no learning).

Our own implementation of the contrast-maximisation idea (Gallego et al., CVPR 2018) with a
piecewise-smooth flow: a grid of flow vectors, one per ``tile_px`` x ``tile_px`` tile, bilinearly
upsampled to a dense (2, H, W) flow in pixels over the window (forward, docs/CONVENTIONS.md).
No code from the copyleft CMax repository excluded by CLAUDE.md rule 6 is used or consulted.

Objective, minimised with Adam (IWE on a canvas padded by ``pad_px``, see ``_contrast``):
    - mean_{t_ref in t_refs} Var(G_sigma * IWE(flow, t_ref)) / Var(G_sigma * IWE(0, t_ref))
    + tv_weight * mean |difference between neighbouring tile vectors|  (px)
IWE = ``evflow.flow.iwe.iwe`` with bilinear splatting (differentiable in the flow), computed by
``iwe_stack`` (one fused scatter, tested equal). Warping to
several reference times (window start, middle, end) is the guard against event collapse (Shiba et
al., Sensors 2022: a flow that collapses events at one t_ref spreads them at the others), the TV
term the second one.

Coarse to fine: stage 1 fits one global translation, stage 2 refines the tiles starting from it;
each stage walks the Gaussian scales ``sigmas`` (large to small) with an equal share of its
iterations. Windows with more than ``max_events`` events are subsampled to evenly spaced events
(deterministic). Hyperparameters are chosen on ``mvsec_tune`` only (configs/cm_baseline.yaml).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from evflow.flow.iwe import normalize_time

MIN_EVENTS = 16  # fewer events than this: zero flow (no contrast to maximise)


@dataclass(frozen=True)
class CMParams:
    tile_px: int = 32
    sigmas: tuple[float, ...] = (3.0, 1.5, 1.0)
    iters_global: int = 60
    iters_tiles: int = 90
    lr: float = 0.3
    tv_weight: float = 0.01
    t_refs: tuple[float, ...] = (0.0, 0.5, 1.0)
    max_events: int = 200_000
    pad_px: int = 64

    def __post_init__(self) -> None:
        if self.tile_px < 1:
            raise ValueError(f"tile_px must be >= 1, got {self.tile_px}")
        if not self.t_refs or not self.sigmas:
            raise ValueError("t_refs and sigmas must be non-empty")
        if (min(self.sigmas) <= 0 or self.lr <= 0 or self.tv_weight < 0 or self.max_events < 1
                or self.pad_px < 0):
            raise ValueError(f"invalid parameters: {self}")

    @classmethod
    def from_dict(cls, d: dict) -> CMParams:
        d = dict(d)
        for k in ("sigmas", "t_refs"):
            if k in d:
                d[k] = tuple(float(v) for v in d[k])
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def tiles_to_flow(tiles: torch.Tensor, *, height: int, width: int, tile_px: int) -> torch.Tensor:
    """(2, Th, Tw) tile vectors -> dense (2, H, W) flow; tile centres at ((i + 0.5) tile_px)."""
    th, tw = tiles.shape[1:]
    up = F.interpolate(tiles[None], size=(th * tile_px, tw * tile_px), mode="bilinear",
                       align_corners=False)
    return up[0, :, :height, :width]


def gauss_blur(imgs: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (R, H, W) images, zero padding."""
    r = max(1, int(math.ceil(3.0 * sigma)))
    k = torch.arange(-r, r + 1, dtype=imgs.dtype, device=imgs.device)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    out = F.conv2d(imgs[:, None], k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(out, k.view(1, 1, -1, 1), padding=(r, 0))[:, 0]


def iwe_stack(x, y, tau, flow, *, t_refs, height: int, width: int, pad: int) -> torch.Tensor:
    """IWEs (R, H + 2 pad, W + 2 pad) of the events warped to each of ``t_refs`` by ``flow``.

    Same definition as ``evflow.flow.iwe.iwe(..., interpolation='bilinear')`` on a canvas shifted
    by ``pad`` (tested equal), written as one scatter over all reference times and taps without
    boolean indexing, so an optimisation step costs a few dozen kernels and no host sync.
    Per-event flow is looked up at the event's own (rounded) pixel, as ``flow.iwe.sample_flow``.
    """
    hc, wc = height + 2 * pad, width + 2 * pad
    xi = x.long() if not torch.is_floating_point(x) else torch.round(x.double()).long()
    yi = y.long() if not torch.is_floating_point(y) else torch.round(y.double()).long()
    pix = yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)
    fe = flow.reshape(2, -1)[:, pix]                                     # (2, N)
    tr = torch.as_tensor(t_refs, dtype=tau.dtype, device=tau.device)[:, None]
    d = tr - tau[None, :]                                                # (R, N)
    xw = x.to(tau.dtype)[None, :] + d * fe[0][None, :] + pad
    yw = y.to(tau.dtype)[None, :] + d * fe[1][None, :] + pad
    x0, y0 = torch.floor(xw), torch.floor(yw)
    fx, fy = xw - x0, yw - y0
    img = torch.zeros(len(t_refs) * hc * wc, dtype=tau.dtype, device=tau.device)
    base = torch.arange(len(t_refs), device=tau.device)[:, None] * (hc * wc)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xt, yt = x0 + dx, y0 + dy
        w = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
        ok = (xt >= 0) & (xt < wc) & (yt >= 0) & (yt < hc)
        idx = base + (yt.clamp(0, hc - 1) * wc + xt.clamp(0, wc - 1)).long()
        img = img.scatter_add(0, idx.reshape(-1), (w * ok).reshape(-1))
    return img.view(len(t_refs), hc, wc)


def _contrast(x, y, tau, flow, *, t_refs, sigma, height, width, pad) -> torch.Tensor:
    # The IWE canvas extends ``pad`` px beyond the sensor on every side: events warped off the
    # sensor still count, otherwise flows that push fewer events out would be favoured.
    imgs = iwe_stack(x, y, tau, flow, t_refs=t_refs, height=height, width=width, pad=pad)
    return gauss_blur(imgs, sigma).flatten(1).var(dim=1).mean()


def _tv(tiles: torch.Tensor) -> torch.Tensor:
    terms = []
    if tiles.shape[2] > 1:
        terms.append((tiles[:, :, 1:] - tiles[:, :, :-1]).abs().mean())
    if tiles.shape[1] > 1:
        terms.append((tiles[:, 1:, :] - tiles[:, :-1, :]).abs().mean())
    return torch.stack(terms).sum() if terms else tiles.new_zeros(())


def _stage(param, to_flow, ev, *, params: CMParams, iters: int, height, width, tv: bool):
    x, y, tau = ev
    opt = torch.optim.Adam([param], lr=params.lr)
    n_s = len(params.sigmas)
    a = dict(t_refs=params.t_refs, height=height, width=width, pad=params.pad_px)
    for si, sigma in enumerate(params.sigmas):
        n_it = iters // n_s + (1 if si < iters % n_s else 0)
        with torch.no_grad():
            c0 = _contrast(x, y, tau, torch.zeros_like(to_flow(param)), sigma=sigma, **a)
        if not c0 > 0:
            return
        for _ in range(n_it):
            opt.zero_grad(set_to_none=True)
            loss = -_contrast(x, y, tau, to_flow(param), sigma=sigma, **a) / c0
            if tv and params.tv_weight > 0:
                loss = loss + params.tv_weight * _tv(param)
            loss.backward()
            opt.step()


def estimate_flow(x, y, t, p, *, t_range, height: int, width: int, params: CMParams,
                  device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Flow (2, H, W) over ``t_range`` (pixels over the window, forward) for one window of
    events. ``p`` is accepted for interface symmetry; the IWE sums both polarities."""
    del p
    n = int(t.numel())
    if n < MIN_EVENTS:
        return torch.zeros((2, height, width), dtype=dtype, device=device)
    if n > params.max_events:
        idx = torch.linspace(0, n - 1, params.max_events, dtype=torch.float64).round().long()
        x, y, t = x[idx], y[idx], t[idx]
    x = x.to(device)
    y = y.to(device)
    tau = normalize_time(t.to(device), t_range, dtype=dtype)
    ev = (x, y, tau)

    g = torch.zeros(2, dtype=dtype, device=device, requires_grad=True)
    _stage(g, lambda v: v.view(2, 1, 1).expand(2, height, width), ev, params=params,
           iters=params.iters_global, height=height, width=width, tv=False)

    th = math.ceil(height / params.tile_px)
    tw = math.ceil(width / params.tile_px)
    tiles = g.detach().view(2, 1, 1).expand(2, th, tw).clone().requires_grad_(True)

    def to_flow(v):
        return tiles_to_flow(v, height=height, width=width, tile_px=params.tile_px)

    _stage(tiles, to_flow, ev, params=params, iters=params.iters_tiles, height=height,
           width=width, tv=True)
    with torch.no_grad():
        return to_flow(tiles).detach().clone()
