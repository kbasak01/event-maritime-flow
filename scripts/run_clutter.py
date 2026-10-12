"""Sea-clutter event-budget study → ``results/maritime/clutter.json`` (Phase 7 step 5; D-058).

Reads the keyframe windows of the 40 ``lars_eval`` clips in each CT variant of the maritime v2e
corpus (``lars_eval_ct067/100/150``, D-054), the LaRS keyframe semantic masks and scene
attributes, and writes per-clip counts, pooled per-class shares / enrichment / events per pixel per
frame interval with cluster-bootstrap CIs, the CT curves, and the attribute breakdown
(``evflow.maritime.clutter``). CPU only.

    python scripts/run_clutter.py
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
from evflow.io.frames import frame_times_us  # noqa: E402
from evflow.maritime import clutter  # noqa: E402
from evflow.sim import maritime_corpus as mc  # noqa: E402
from evflow.sim.v2e_runner import git_dirty, git_sha  # noqa: E402
from evflow.train.data import EventArrays  # noqa: E402

OUT = REPO / "results" / "maritime" / "clutter.json"
SIZE = (640, 360)


def main(argv: list[str] | None = None) -> int:
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    cfg = yaml.safe_load((REPO / "configs/v2e_corpus_maritime.yaml").read_text())
    plan = mc.load_plan(paths)
    root = mc.out_root(paths)
    variants = sorted(float(m) for m in cfg["ct_variants"])
    attrs_all = lars_io.load_attributes(paths["lars"]["root"], "val")
    by_ct, clip_rows = {}, {}
    order: list[str] | None = None
    for m in variants:
        corpus = f"lars_eval_{mc.ct_tag(m)}"
        units = [u for u in plan["units"] if u["corpus"] == corpus]
        if order is None:
            order = [u["clip"] for u in units]
        elif [u["clip"] for u in units] != order:
            raise SystemExit(f"{corpus}: clips differ from the other CT variants")
        clips, rows = [], []
        for u in units:
            if u["split"] != "lars_eval":
                raise SystemExit(f"{u['key']}: not a lars_eval unit")
            man = json.loads((root / corpus / f"{u['name']}.json").read_text())
            ft = frame_times_us(int(u["n_frames"]), float(man["input_frame_rate"]),
                                int(man["conversion"]["t_offset_us"]))
            ks = [j - 1 for j in u["keyframes_local"]]
            arr = EventArrays.from_store(root / corpus / f"{u['name']}.h5", int(ft[min(ks)]),
                                         int(ft[max(ks) + 1]), known_end_us=int(ft[-1]))
            counts = []
            for kf, k in zip(u["keyframes"], ks, strict=True):
                ev = arr.window(int(ft[k]), int(ft[k + 1]))
                masks = lars_io.class_masks(lars_io.semantic_mask(
                    paths["lars"]["root"], u["subset"], kf, size=SIZE))
                counts.append(clutter.window_class_counts(ev["x"], ev["y"], masks))
            st = clutter.clip_stats(counts)
            st.update(clip=u["clip"], keyframes=u["keyframes"], n_windows=len(counts),
                      store_sha256=man["store_sha256"], end_clamped_us=arr.end_clamped_us)
            clips.append(st)
            rows.append({"clip": u["clip"], "n_events": st["n_events"],
                         **{c: st["classes"][c] for c in lars_io.CLASSES}})
        by_ct[mc.ct_tag(m)] = {"ct_multiplier": m, "corpus": corpus,
                               "pos_neg_thres": float(units[0]["overrides"].get(
                                   "--pos_thres", 0)) or None,
                               "n_clips": len(clips), "pooled": clutter.pooled(clips)}
        clip_rows[mc.ct_tag(m)] = (clips, rows)

    base_clips = clip_rows[mc.ct_tag(1.0)][0]
    attrs = []
    for c in base_clips:
        kf = c["keyframes"][0]          # a 2-keyframe clip would use the first (none here)
        attrs.append(attrs_all[kf])
    breakdown = clutter.attribute_breakdown(base_clips, attrs)
    ct_ratio = {}
    for c in lars_io.CLASSES:
        e = {t: np.array([r["classes"][c]["events_per_px"] for r in clip_rows[t][0]])
             for t in clip_rows}
        ct_ratio[c] = {f"{t}_over_ct100": float(np.nanmedian(e[t] / e["ct100"]))
                       for t in e if t != "ct100"}
    doc = {
        "kind": "clutter_study", "tier": "T3b",
        "tier_note": "synthetic: v2e events simulated from LaRS frames (assumed 30 fps, D-024; "
                     "640x360 stretched, D-054); event-budget statistics, no flow reference; "
                     "inherits the source frames' dynamic range and blur (plan section 0(1))",
        "plan_step": "Phase 7 step 5",
        "protocol": {"windows": "keyframe window (keyframe - 1 -> keyframe), one frame interval",
                     "labels": "LaRS keyframe semantic mask, nearest-resized to 640x360; "
                               "ignore px count in totals, in no class",
                     "shares": "event_share = class events / all events; pixel_share = class "
                               "px / all px; enrichment = event_share / pixel_share; "
                               "events_per_px per frame interval",
                     "pooled": "events and pixels summed over clips; 95 % CI = cluster "
                               "bootstrap over clips, 1000 resamples, seed 0",
                     "attributes": "LaRS image_annotations.json of the (first) keyframe; "
                                   f"levels with n < {clutter.MIN_N} clips not reported; "
                                   "Spearman rho with clip-bootstrap CI at CT x1.0",
                     "ct_variants": variants},
        "n_clips": len(base_clips),
        "by_ct": by_ct,
        "events_per_px_ratio_vs_ct100_median_over_clips": ct_ratio,
        "attributes_ct100": breakdown,
        "clips_ct100": clip_rows["ct100"][1],
        "git_sha": git_sha(), "git_dirty": git_dirty(), "created_by": "scripts/run_clutter.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    w = by_ct["ct100"]["pooled"]["water"]
    print(f"clutter: {len(base_clips)} clips; water event share {w['event_share']['mean']:.3f} "
          f"vs pixel share {w['pixel_share']['mean']:.3f} at CT x1.0 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
