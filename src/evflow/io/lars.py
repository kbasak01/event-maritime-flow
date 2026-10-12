"""LaRS reader (IMPLEMENTATION_PLAN §1.2 and Phase 7 steps 1 and 5).

Layout (``paths.local.yaml`` ``lars.root``, D-011): ``<subset>/images_seq/<prefix>_<nnnnn>.jpg``
(context frames + keyframes), ``<subset>/semantic_masks/<keyframe>.png``,
``<subset>/image_annotations.json`` (scene attributes per keyframe) and
``<subset>/panoptic_annotations.json``. Clips are the maximal runs of consecutive frames listed in
``splits/lars_*.json`` (D-023).

- **Frame rate**: LaRS documents none; 30 fps is assumed (D-024, ``configs/v2e_profiles.yaml``
  ``lars.assumed_fps``). Flow is reported per frame interval, never per second.
- **Class masks**: semantic masks are single-channel uint8; the value → class mapping
  (``SEMANTIC_VALUES``) is checked against the panoptic annotation areas by
  ``verify_class_mapping`` rather than assumed. 255 = ignore.
- **Attributes**: only the fields present in ``image_annotations.json`` (inspected 2026-10-12):
  ``scene_type``, ``lighting``, ``reflections``, ``waves`` and the ``special.*`` booleans.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import numpy as np

from evflow.io.frames import native_size, resize_labels  # noqa: F401 (re-export)

CLASSES = ("water", "sky", "obstacle")
SEMANTIC_VALUES = {"obstacle": 0, "water": 1, "sky": 2}
IGNORE_VALUE = 255
# panoptic supercategory → class (categories of panoptic_annotations.json)
SUPERCATEGORY = {"obstacle": "obstacle", "water": "water", "sky": "sky"}
ATTRIBUTE_FIELDS = ("scene_type", "lighting", "reflections", "waves")
SPECIAL_FIELDS = ("extra_dark", "extra_bright", "glitter", "dirty_lens", "wakes", "rain", "fog",
                  "plants_debris")
# ordinal attributes (levels in increasing order) for rank correlations
ORDINAL = {"waves": ("still", "moderate", "rough"),
           "reflections": ("none", "moderate", "heavy")}


# ---------------------------------------------------------------------------------------------
# clips and frames

def every_kth(clips: list[dict], n: int) -> list[dict]:
    """D-026 rule: every k-th clip of the split in manifest order, k = len // n, first n."""
    if n <= 0 or n > len(clips):
        raise ValueError(f"cannot take {n} of {len(clips)} clips")
    return clips[:: len(clips) // n][:n]


def clip_frames(root: str | PathLike, subset: str, prefix: str, first: int,
                last: int) -> list[Path]:
    """``images_seq`` files of frames ``first..last`` (inclusive); all must exist."""
    d = Path(root) / subset / "images_seq"
    files = [d / f"{prefix}_{n:05d}.jpg" for n in range(first, last + 1)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"LaRS clip frames missing: {missing[:3]}")
    return files


def keyframe_indices(clip: dict) -> list[int]:
    """Local frame indices (0 = ``clip['first']``) of the clip's keyframes."""
    out = []
    for kf in clip["keyframes"]:
        n = int(kf.rsplit("_", 1)[1])
        if not clip["first"] <= n <= clip["last"]:
            raise ValueError(f"{kf} outside clip {clip['clip']}")
        out.append(n - int(clip["first"]))
    return out


def clip_spec(clip: dict) -> str:
    """Runner argument ``prefix:first:last``."""
    return f"{clip['prefix']}:{int(clip['first'])}:{int(clip['last'])}"


def read_frame_bgr(path: str | PathLike) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im


# ---------------------------------------------------------------------------------------------
# annotations

def load_attributes(root: str | PathLike, subset: str) -> dict[str, dict]:
    """keyframe stem → flat attributes (``special.*`` flattened); values kept as stored
    (``None`` where the annotation has null)."""
    doc = json.loads((Path(root) / subset / "image_annotations.json").read_text())
    out = {}
    for a in doc["annotations"]:
        lab = a["labels"]
        row = {k: lab.get(k) for k in ATTRIBUTE_FIELDS}
        sp = lab.get("special") or {}
        for k in SPECIAL_FIELDS:
            v = sp.get(k)
            row[f"special.{k}"] = None if v is None else bool(v)
        out[Path(a["file_name"]).stem] = row
    return out


def semantic_mask(root: str | PathLike, subset: str, keyframe: str,
                  size: tuple[int, int] | None = None) -> np.ndarray:
    """uint8 label map of a keyframe, optionally resized (nearest) to ``size = (W, H)``."""
    from PIL import Image

    with Image.open(Path(root) / subset / "semantic_masks" / f"{keyframe}.png") as im:
        lab = np.array(im)
    if lab.ndim != 2:
        raise ValueError(f"{keyframe}: semantic mask is not single-channel ({lab.shape})")
    return resize_labels(lab, size) if size is not None else lab


def class_masks(lab: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean map per class; ``ignore`` holds 255 and any value outside the mapping."""
    out = {c: lab == v for c, v in SEMANTIC_VALUES.items()}
    known = np.zeros(lab.shape, bool)
    for m in out.values():
        known |= m
    out["ignore"] = ~known
    return {c: out[c] for c in (*CLASSES, "ignore")}


def verify_class_mapping(root: str | PathLike, subset: str, keyframes: list[str] | None = None,
                         tol: float = 0.02) -> dict:
    """Check ``SEMANTIC_VALUES`` against ``panoptic_annotations.json``: for each keyframe the
    pixel count of each semantic value must equal the summed panoptic area of the matching
    supercategory within ``tol`` (relative to the image area). Raises on any mismatch."""
    from PIL import Image

    doc = json.loads((Path(root) / subset / "panoptic_annotations.json").read_text())
    sup = {c["id"]: SUPERCATEGORY[c["supercategory"]] for c in doc["categories"]}
    by_file = {Path(a["file_name"]).stem: a for a in doc["annotations"]}
    names = keyframes if keyframes is not None else sorted(by_file)
    worst = {c: 0.0 for c in CLASSES}
    for kf in names:
        ann = by_file[kf]
        area = {c: 0 for c in CLASSES}
        for s in ann["segments_info"]:
            area[sup[s["category_id"]]] += int(s["area"])
        with Image.open(Path(root) / subset / "semantic_masks" / f"{kf}.png") as im:
            lab = np.array(im)
        n = lab.size
        for c, v in SEMANTIC_VALUES.items():
            d = abs(int((lab == v).sum()) - area[c]) / n
            worst[c] = max(worst[c], d)
            if d > tol:
                raise ValueError(f"{subset}/{kf}: semantic value {v} has {(lab == v).sum()} px, "
                                 f"panoptic '{c}' area {area[c]} (|diff| {d:.3f} of image)")
    return {"subset": subset, "n_keyframes": len(names), "mapping": dict(SEMANTIC_VALUES),
            "ignore_value": IGNORE_VALUE, "max_abs_area_diff_fraction": worst, "tol": tol}
