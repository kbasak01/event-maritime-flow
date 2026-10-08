"""DSEC 16-bit optical-flow PNG codec.

DSEC-Flow GT (uzh-rpg/DSEC ``check_optical_flow_submission.py``): RGB uint16 PNG with
``u = (R - 2**15) / 128``, ``v = (G - 2**15) / 128``, valid where ``B == 1``. Flow is forward, in
pixels over the GT interval (100 ms for DSEC-Flow). OpenCV stores channels as BGR; the reversal
happens here and nowhere else.
"""

from __future__ import annotations

from os import PathLike

import cv2
import numpy as np

_OFFSET = 2**15
_SCALE = 128.0


def decode_dsec_flow(rgb16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) uint16 in R, G, B order -> flow (2, H, W) float64, valid (H, W) bool."""
    rgb16 = np.asarray(rgb16)
    if rgb16.ndim != 3 or rgb16.shape[2] != 3 or rgb16.dtype != np.uint16:
        raise ValueError(f"expected (H, W, 3) uint16, got {rgb16.shape} {rgb16.dtype}")
    flow = (rgb16[..., :2].astype(np.float64) - _OFFSET) / _SCALE
    return np.moveaxis(flow, -1, 0), rgb16[..., 2] == 1


def encode_dsec_flow(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """flow (2, H, W) px, valid (H, W) -> (H, W, 3) uint16 in R, G, B order; invalid pixels = 0."""
    flow = np.asarray(flow, np.float64)
    valid = np.asarray(valid, bool)
    q = np.round(flow * _SCALE) + _OFFSET
    if np.any((q[:, valid] < 0) | (q[:, valid] > 65535)):
        raise ValueError("flow outside the 16-bit PNG range (about +-256 px)")
    rgb = np.zeros((*valid.shape, 3), np.uint16)
    rgb[valid, 0] = q[0][valid]
    rgb[valid, 1] = q[1][valid]
    rgb[valid, 2] = 1
    return rgb


def read_dsec_flow_png(path: str | PathLike) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return decode_dsec_flow(bgr[..., ::-1].copy())


def write_dsec_flow_png(path: str | PathLike, flow: np.ndarray, valid: np.ndarray) -> None:
    if not cv2.imwrite(str(path), encode_dsec_flow(flow, valid)[..., ::-1].copy()):
        raise OSError(f"could not write {path}")
