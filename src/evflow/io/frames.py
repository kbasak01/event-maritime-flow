"""Frame geometry and clock shared by the maritime readers (Phase 7; D-054).

v2e resizes every input frame with ``cv2.resize(frame, (W, H), interpolation=INTER_AREA)``
straight to the profile's output size (external/v2e/v2e.py, before its BGR→luma step), so a
non-16:9 LaRS frame is stretched anisotropically (user decision, D-054). Everything that must
share the event geometry — RAFT teacher frames, class masks, qualitative panels — is resized here
the same way: images with INTER_AREA, label maps with INTER_NEAREST.

v2e's clock for frame k of a folder input is ``k / fps`` seconds from the first frame
(``sim.v2e_runner.uniform_clock_us``); a window spans source frames k → k+1 and its integer µs
bounds are the rounded frame times plus the store's ``t_offset``.
"""

from __future__ import annotations

import numpy as np


def resize_like_v2e(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``size = (width, height)``; INTER_AREA, as v2e does (no-op when already that size)."""
    import cv2

    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def resize_labels(lab: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label map to ``size = (width, height)``."""
    import cv2

    w, h = size
    if lab.shape[1] == w and lab.shape[0] == h:
        return lab
    return cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)


def scale_factors(native: tuple[int, int], size: tuple[int, int]) -> tuple[float, float]:
    """Output px per native px along x and y (equal only for an aspect-preserving resize)."""
    return size[0] / native[0], size[1] / native[1]


def native_size(path) -> tuple[int, int]:
    """(width, height) of an image file without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def frame_times_us(n_frames: int, fps: float, t_offset_us: int = 0) -> np.ndarray:
    """Absolute integer µs of source frames 0..n-1 on v2e's uniform clock."""
    return int(t_offset_us) + np.round(np.arange(n_frames, dtype=np.float64)
                                       * (1e6 / fps)).astype(np.int64)
