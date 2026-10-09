"""Evidence for the LaRS frame-rate assumption (D-024): inter-frame motion, LaRS vs SMD at 30 fps.

    python scripts/lars_fps_check.py   # -> results/lars_fps_check.json

LaRS documents no frame rate for ``images_seq``. If its context frames were subsampled from video,
inter-frame motion would exceed that of comparable on-board video at its native rate. Median
Farnebäck flow magnitude (both resized to 640×360) over 40 lars_train clips (3 pairs each) and
the 8 smd_train videos (4 pairs each, SMD is 30 fps by ffprobe). Train splits only.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def flow_median(a: np.ndarray, b: np.ndarray) -> float:
    a, b = cv2.resize(a, (640, 360)), cv2.resize(b, (640, 360))
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(np.hypot(f[..., 0], f[..., 1])))


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    clips = json.loads((REPO / "splits/lars_train.json").read_text())["clips"]
    clips = clips[:: len(clips) // 40][:40]
    d = Path(cfg["lars"]["root"]) / "train" / "images_seq"
    lars = []
    for c in clips:
        fr = [cv2.imread(str(d / f"{c['prefix']}_{n:05d}.jpg"), 0)
              for n in range(c["first"], c["first"] + 4)]
        lars += [flow_median(fr[i], fr[i + 1]) for i in range(3)]
    smd = []
    for v in json.loads((REPO / "splits/smd_train.json").read_text())["videos"]:
        cap = cv2.VideoCapture(str(Path(cfg["smd"]["onboard"]) / "Videos" / f"{v}.avi"))
        for i in range(60):
            ok, im = cap.read()
            if not ok:
                break
            if i % 15 == 0:
                ok2, im2 = cap.read()
                if ok2:
                    smd.append(flow_median(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                                           cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    doc = {"kind": "lars_fps_check", "decision": "D-024", "tier": "n/a",
           "tier_note": "frame-rate heuristic for a v2e input setting, not an evidence-tier result",
           "lars_pairs": len(lars), "smd_pairs": len(smd),
           "lars_median_px_per_frame": float(np.median(lars)),
           "lars_iqr": [float(np.percentile(lars, 25)), float(np.percentile(lars, 75))],
           "smd30fps_median_px_per_frame": float(np.median(smd)),
           "smd30fps_iqr": [float(np.percentile(smd, 25)), float(np.percentile(smd, 75))],
           "git_sha": sha, "created_by": "scripts/lars_fps_check.py",
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (REPO / "results/lars_fps_check.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
