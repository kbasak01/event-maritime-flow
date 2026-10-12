"""Near-duplicate check: LaRS clips taken from SMD footage vs the SMD on-board videos (Phase 7;
event-data-auditor finding 1, D-061) → ``results/maritime/lars_smd_overlap.json``.

LaRS includes clips whose prefix starts with ``smd`` (SMD-derived footage). If any of them showed
the same frames as an SMD on-board video, a model trained on one split could be evaluated on
content it saw (e.g. a ``lars_train`` clip copying an ``smd_test`` video). Every frame of every
``smd*`` LaRS clip in the maritime corpus is reduced to a 64×36 grey thumbnail (INTER_AREA) and
compared by mean squared error (8-bit units²) with every frame of the 11 on-board videos. The
reference scale is the smallest frame-to-frame MSE between two *different* on-board videos (the
on-board videos are distinct recordings): a LaRS clip is flagged as a possible copy when its best
match is below that. CPU only; reads frames, writes no image.

    python scripts/check_lars_smd_overlap.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

from evflow.io import lars as lars_io  # noqa: E402
from evflow.io import smd as smd_io  # noqa: E402
from evflow.sim import maritime_corpus as mc  # noqa: E402
from evflow.sim.v2e_runner import git_dirty, git_sha  # noqa: E402

OUT = REPO / "results" / "maritime" / "lars_smd_overlap.json"
THUMB = (64, 36)


def thumb(im: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    import cv2

    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    videos = {}
    for split in ("smd_train", "smd_test"):
        for v in json.loads((REPO / f"splits/{split}.json").read_text())["videos"]:
            cap = cv2.VideoCapture(str(smd_io.video_path(paths["smd"]["onboard"], v)))
            fr = []
            while True:                       # stream: thumbnails only, never full frames
                ok, im = cap.read()
                if not ok:
                    break
                fr.append(thumb(im))
            cap.release()
            videos[(split, Path(v).stem)] = np.stack(fr)
    keys = list(videos)
    cross = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va, vb = videos[a][::5], videos[b][::5]
            d = ((va[:, None] - vb[None]) ** 2).mean(axis=(2, 3))
            cross.append({"a": a[1], "b": b[1], "min_mse": float(d.min())})
    ref = min(c["min_mse"] for c in cross)
    plan = mc.load_plan(paths)
    rows = []
    for u in plan["units"]:
        if u["source"] != "lars" or not u["clip"].startswith("smd") or u["corpus"] not in (
                "lars_train", "lars_eval_ct100"):
            continue
        prefix, first, last = u["clip"].rsplit(":", 2)
        files = lars_io.clip_frames(paths["lars"]["root"], u["subset"], prefix, int(first),
                                    int(last))
        th = np.stack([thumb(cv2.imread(str(f), cv2.IMREAD_COLOR)) for f in files])
        best = (np.inf, None, None)
        for k, vv in videos.items():
            d = ((th[:, None] - vv[None]) ** 2).mean(axis=(2, 3))
            if d.min() < best[0]:
                j = int(d.min(axis=0).argmin())
                best = (float(d.min()), k, j)
        rows.append({"corpus": u["corpus"], "split": u["split"], "clip": u["clip"],
                     "n_frames": len(files), "best_mse": best[0],
                     "nearest_video": best[1][1], "nearest_video_split": best[1][0],
                     "nearest_video_frame": best[2], "possible_copy": best[0] < ref})
    doc = {
        "kind": "data_overlap_check", "tier": "n/a",
        "tier_note": "data-integrity check (no metrics): LaRS smd* clips of the maritime corpus "
                     "vs the SMD on-board videos",
        "method": f"{THUMB[0]}x{THUMB[1]} grey thumbnails (INTER_AREA), MSE in 8-bit units^2; "
                  "every clip frame vs every on-board frame; reference = smallest MSE between "
                  "two different on-board videos (every 5th frame); flag if best < reference",
        "reference_min_cross_video_mse": ref, "cross_video_pairs": cross,
        "n_clips_checked": len(rows), "n_possible_copies": sum(r["possible_copy"] for r in rows),
        "clips": rows,
        "git_sha": git_sha(), "git_dirty": git_dirty(),
        "created_by": "scripts/check_lars_smd_overlap.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(rows)} LaRS smd* clips; reference cross-video MSE {ref:.1f}; best clip MSE "
          f"{min(r['best_mse'] for r in rows):.1f}; possible copies "
          f"{doc['n_possible_copies']} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
