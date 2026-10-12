"""RAFT teacher for the maritime track (IMPLEMENTATION_PLAN §1.6 teacher-EPE, Phase 7 step 2).

torchvision ``raft_large`` (``Raft_Large_Weights.C_T_SKHT_V2``, downloaded at runtime and never
redistributed, docs/LICENSES.md) runs on the *source frames* of each v2e unit, resized exactly as
v2e resized them (``io.frames.resize_like_v2e``: INTER_AREA to 640×360, aspect not kept for LaRS,
D-054), forward (k → k+1) and backward (k+1 → k). The forward flow of the pair is the teacher for
the event window spanning frames k → k+1 (px per frame interval, forward, ``(2, H, W)``).

Forward–backward consistency (§1.6; squared form of Sundaram et al. 2010 / UnFlow, D-055): a pixel
x is consistent when x + f(x) lies inside the image and

    |f(x) + b(x + f(x))|²  <  alpha1 · (|f(x)|² + |b(x + f(x))|²) + alpha2

with ``b`` sampled bilinearly, alpha1 = 0.01 and alpha2 = 0.5 (``configs/evflow.yaml``
``metrics.teacher``). The teacher is a frame-based estimate, **not** ground truth: numbers scored
against it are T3a "teacher-EPE".
"""

from __future__ import annotations

import numpy as np

WEIGHTS = "C_T_SKHT_V2"
ITERS = 12            # torchvision raft_large default num_flow_updates


# ---------------------------------------------------------------------------------------------
# consistency (numpy reference + torch)

def _bilinear_ref(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample (C, H, W) at float coordinates; coordinates must lie in [0, W-1] × [0, H-1]."""
    _, h, w = img.shape
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 2 if w > 1 else 0)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 2 if h > 1 else 0)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = xs - x0, ys - y0
    return (img[:, y0, x0] * (1 - ax) * (1 - ay) + img[:, y0, x1] * ax * (1 - ay)
            + img[:, y1, x0] * (1 - ax) * ay + img[:, y1, x1] * ax * ay)


def fb_consistency_mask_ref(f: np.ndarray, b: np.ndarray, alpha1: float,
                            alpha2: float) -> np.ndarray:
    """Pure-numpy reference: ``f``, ``b`` are (2, H, W) forward/backward flows → bool (H, W)."""
    f = np.asarray(f, np.float64)
    b = np.asarray(b, np.float64)
    _, h, w = f.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xt, yt = xs + f[0], ys + f[1]
    inside = (xt >= 0) & (xt <= w - 1) & (yt >= 0) & (yt <= h - 1)
    bw = _bilinear_ref(b, np.clip(xt, 0, w - 1), np.clip(yt, 0, h - 1))
    lhs = (f[0] + bw[0]) ** 2 + (f[1] + bw[1]) ** 2
    rhs = alpha1 * (f[0] ** 2 + f[1] ** 2 + bw[0] ** 2 + bw[1] ** 2) + alpha2
    return inside & (lhs < rhs)


def fb_consistency_mask_literal_ref(f: np.ndarray, b: np.ndarray, alpha1: float,
                                    alpha2: float) -> np.ndarray:
    """§1.6 as printed (unsquared left side); reported as a coverage only (D-055)."""
    f = np.asarray(f, np.float64)
    b = np.asarray(b, np.float64)
    _, h, w = f.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xt, yt = xs + f[0], ys + f[1]
    inside = (xt >= 0) & (xt <= w - 1) & (yt >= 0) & (yt <= h - 1)
    bw = _bilinear_ref(b, np.clip(xt, 0, w - 1), np.clip(yt, 0, h - 1))
    lhs = np.sqrt((f[0] + bw[0]) ** 2 + (f[1] + bw[1]) ** 2)
    rhs = alpha1 * (f[0] ** 2 + f[1] ** 2 + bw[0] ** 2 + bw[1] ** 2) + alpha2
    return inside & (lhs < rhs)


def fb_consistency_mask(f, b, alpha1: float, alpha2: float):
    """Torch version for batches: ``f``, ``b`` (N, 2, H, W) → bool (N, H, W). Equal to the
    reference (bilinear ``grid_sample``, align_corners=True, i.e. pixel centres at integers)."""
    import torch
    import torch.nn.functional as F

    n, _, h, w = f.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=f.dtype, device=f.device),
                            torch.arange(w, dtype=f.dtype, device=f.device), indexing="ij")
    xt, yt = xs + f[:, 0], ys + f[:, 1]
    inside = (xt >= 0) & (xt <= w - 1) & (yt >= 0) & (yt <= h - 1)
    gx = 2 * xt.clamp(0, w - 1) / max(w - 1, 1) - 1
    gy = 2 * yt.clamp(0, h - 1) / max(h - 1, 1) - 1
    bw = F.grid_sample(b, torch.stack([gx, gy], dim=-1), mode="bilinear",
                       padding_mode="border", align_corners=True)
    lhs = ((f + bw) ** 2).sum(1)
    rhs = alpha1 * ((f ** 2).sum(1) + (bw ** 2).sum(1)) + alpha2
    return inside & (lhs < rhs)


# ---------------------------------------------------------------------------------------------
# network

def pad_to_multiple(x, k: int = 8):
    """Replicate-pad (N, C, H, W) on the bottom/right to multiples of ``k``; returns (x, (h, w))."""
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ph, pw = (-h) % k, (-w) % k
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


class RaftTeacher:
    """torchvision RAFT on uint8 RGB frames (H, W, 3). ``model`` may be injected (tests)."""

    def __init__(self, device: str = "cuda", weights: str = WEIGHTS, iters: int = ITERS,
                 model=None):
        import torch

        self.device = torch.device(device)
        self.iters = int(iters)
        self.weights = weights
        if model is None:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            w = getattr(Raft_Large_Weights, weights)
            model = raft_large(weights=w, progress=False)
            self.weights_url = w.url
        else:
            self.weights_url = None
        self.model = model.eval().to(self.device)

    @staticmethod
    def _prep(frames: list[np.ndarray], device):
        import torch

        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        return (x / 255.0 * 2.0 - 1.0).to(device)

    def __call__(self, img1: list[np.ndarray], img2: list[np.ndarray]) -> np.ndarray:
        """Forward flow img1 → img2 for a batch, (N, 2, H, W) float32."""
        import torch

        a, (h, w) = pad_to_multiple(self._prep(img1, self.device))
        b, _ = pad_to_multiple(self._prep(img2, self.device))
        with torch.inference_mode():
            flow = self.model(a, b, num_flow_updates=self.iters)[-1]
        return flow[:, :, :h, :w].float().cpu().numpy()

    def pairs(self, frames: list[np.ndarray], alpha1: float, alpha2: float,
              batch: int = 4) -> dict:
        """Teacher for every consecutive pair of RGB ``frames``: forward flow (n-1, 2, H, W),
        consistency mask (n-1, H, W) and the literal-§1.6 mask (coverage only)."""
        import torch

        fw, ok, lit = [], [], []
        for s in range(0, len(frames) - 1, batch):
            e = min(s + batch, len(frames) - 1)
            a, b = frames[s:e], frames[s + 1:e + 1]
            f = self(a, b)
            r = self(b, a)
            ft, rt = torch.from_numpy(f), torch.from_numpy(r)
            ok.append(fb_consistency_mask(ft.double(), rt.double(), alpha1, alpha2).numpy())
            lit.append(np.stack([fb_consistency_mask_literal_ref(f[i], r[i], alpha1, alpha2)
                                 for i in range(len(f))]))
            fw.append(f)
        return {"flow": np.concatenate(fw), "valid": np.concatenate(ok),
                "valid_literal": np.concatenate(lit)}
