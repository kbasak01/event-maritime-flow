"""Split guard for v2e inputs (Phase 2 gate open item: the runner checks every MVSEC APS range
against a named split manifest before converting it).

An MVSEC split manifest (``splits/mvsec_*.json``) names the APS frames a job may feed to v2e:

- ``mvsec_train`` / ``mvsec_tune``: the frames of each ``aps[]`` range (one sequence);
- ``mvsec_test`` (evaluation only): per sequence, the first window start to the last frame any
  window uses (``aps_last + dt``). A test unit may start up to ``MAX_LEAD_FRAMES`` earlier inside
  the same sequence, so v2e state settles before the first owned window (D-032). Lead frames are
  refused for training/tuning splits: they would reach into the gaps between them.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SPLITS = REPO / "splits"
MAX_LEAD_FRAMES = 10          # = calibration_grid.warmup_frames (D-027)
EVAL_SPLITS = ("mvsec_test",)


class LeakageError(ValueError):
    """A v2e input or statistics window outside the split it claims."""


def load_split(name: str, root: Path = SPLITS) -> dict:
    doc = json.loads((root / f"{name}.json").read_text())
    if doc.get("split") != name:
        raise LeakageError(f"splits/{name}.json declares split={doc.get('split')!r}")
    return doc


def split_aps_bounds(split: dict, sequence: str) -> list[tuple[int, int]]:
    """Inclusive APS frame ranges of ``sequence`` that ``split`` may feed to v2e."""
    if split.get("dataset") != "mvsec":
        raise LeakageError(f"split {split.get('split')!r} is not an MVSEC split")
    if "sequences" in split:
        for s in split["sequences"]:
            if s["sequence"] == sequence:
                w = s["windows"]
                first = min(int(v["aps_first"]) for v in w.values())
                last = max(int(v["aps_last"]) + int(k.removeprefix("dt")) for k, v in w.items())
                return [(first, last)]
        raise LeakageError(f"{sequence} is not in split {split['split']}")
    if split.get("sequence") != sequence:
        raise LeakageError(f"{sequence} is not the sequence of split {split['split']}")
    return [(int(a["aps_first"]), int(a["aps_last"])) for a in split["aps"]]


def check_aps_in_split(split: dict, sequence: str, first: int, last: int,
                       lead_frames: int = 0) -> None:
    """Raise unless APS frames ``first..last`` (inclusive) lie inside one range of the split
    (the start may precede it by ``lead_frames`` for evaluation splits only)."""
    if not 0 <= lead_frames <= MAX_LEAD_FRAMES:
        raise LeakageError(f"lead_frames={lead_frames} outside [0, {MAX_LEAD_FRAMES}]")
    if lead_frames and split.get("split") not in EVAL_SPLITS:
        raise LeakageError(f"lead frames are allowed for {EVAL_SPLITS} only, "
                           f"not {split.get('split')}")
    if first > last:
        raise LeakageError(f"empty APS range [{first}, {last}]")
    bounds = split_aps_bounds(split, sequence)
    if not any(max(lo - lead_frames, 0) <= first and last <= hi for lo, hi in bounds):
        raise LeakageError(f"{sequence} APS [{first}, {last}] (lead {lead_frames}) is outside "
                           f"{split.get('split')} {bounds}")
