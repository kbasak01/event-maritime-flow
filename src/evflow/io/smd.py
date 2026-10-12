"""SMD (Singapore Maritime Dataset, VIS on-board) reader (IMPLEMENTATION_PLAN §1.2, Phase 7).

Videos live in ``paths.local.yaml`` ``smd.onboard`` + ``/Videos/<name>.avi`` (30 fps colour
1920×1080 AVIs). v2e is fed losslessly decoded PNG folders, never the AVI (D-025). The Phase 7
segments (D-054, superseding the SMD line of D-026; most videos are 8.5–10 s long):
``smd_train`` = every train video over [0, 5.6] s, ``smd_test`` = every test video over
[0, min(10 s, last frame)]. No SMD imagery may be shown (docs/LICENSES.md, D-008).
"""

from __future__ import annotations

import subprocess
from os import PathLike
from pathlib import Path

import numpy as np

from evflow.io.frames import resize_like_v2e


def video_path(onboard_root: str | PathLike, name: str) -> Path:
    stem = Path(name).stem
    return Path(onboard_root) / "Videos" / f"{stem}.avi"


def probe(video: str | PathLike) -> dict:
    """fps (ffprobe ``r_frame_rate``) and frame count (``nb_frames``) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,nb_frames,width,height", "-of", "default=nw=1", str(video)],
        capture_output=True, text=True, check=True).stdout
    kv = dict(line.split("=", 1) for line in out.strip().splitlines())
    num, den = kv["r_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "n_frames": int(kv["nb_frames"]),
            "width": int(kv["width"]), "height": int(kv["height"]),
            "r_frame_rate": kv["r_frame_rate"]}


def segment(start_s: float, stop_s: float, fps: float, n_frames: int) -> dict:
    """Segment rule: frames ``round(start·fps) .. min(round(stop·fps), n_frames − 1)``; the stop
    passed to the runner is the last frame's time, so the runner reproduces the same frames."""
    f0 = int(round(start_s * fps))
    f1 = min(int(round(stop_s * fps)), n_frames - 1)
    if f1 <= f0:
        raise ValueError(f"empty segment [{start_s}, {stop_s}] s of a {n_frames}-frame video")
    return {"first_frame": f0, "last_frame": f1, "start_s": f0 / fps, "stop_s": f1 / fps,
            "n_frames": f1 - f0 + 1, "video_s": round((f1 - f0) / fps, 4)}


def read_frames(video: str | PathLike, first: int, last: int,
                size: tuple[int, int] | None = None) -> list[np.ndarray]:
    """BGR frames ``first..last`` decoded with ``cv2.VideoCapture`` (the decoder that wrote the
    v2e input PNGs, ``sim.v2e_runner.export_video_frames``), optionally resized like v2e."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        for _ in range(first):
            if not cap.grab():
                raise ValueError(f"{video}: ended before frame {first}")
        out = []
        for k in range(last - first + 1):
            ok, im = cap.read()
            if not ok:
                raise ValueError(f"{video}: ended at frame {first + k}")
            out.append(resize_like_v2e(im, size) if size is not None else im)
    finally:
        cap.release()
    return out
