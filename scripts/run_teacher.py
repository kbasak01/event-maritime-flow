"""RAFT teacher flows for the maritime corpus (IMPLEMENTATION_PLAN Phase 7 step 2; D-055).

For every unit of the chosen corpora (from ``<events>/v2e/maritime/corpus_plan.json``) the v2e
input frames are re-read (SMD: the lossless PNG cache v2e was fed; LaRS: the clip JPGs), checked
against the unit manifest's input SHA-256, resized exactly as v2e resized them, and passed through
torchvision RAFT forward and backward. Writes ``<events>/teacher/<corpus>/<unit>.h5`` (skipped if
present) and ``results/teacher/summary.json`` (coverage per unit; per LaRS class on keyframe
windows). The LaRS CT variants share their frames, so only ``lars_eval_ct100`` gets a teacher.

    python scripts/run_teacher.py                      # all teacher corpora, then summarise
    python scripts/run_teacher.py --corpus smd_test --limit 1    # smoke
    python scripts/run_teacher.py --summarise-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evflow.io import lars as lars_io  # noqa: E402
from evflow.io.frames import resize_like_v2e  # noqa: E402
from evflow.sim import maritime_corpus as mc  # noqa: E402
from evflow.sim.v2e_runner import git_dirty, git_sha, gpu_name, sha256_frames  # noqa: E402
from evflow.teacher import raft  # noqa: E402
from evflow.teacher.cache import TeacherCache, write  # noqa: E402

CORPORA = ("smd_train", "smd_test", "lars_train", "lars_eval_ct100")
RESULT = REPO / "results" / "teacher" / "summary.json"


def teacher_root(paths: dict) -> Path:
    return Path(paths["outputs"]["events"]) / "teacher"


def teacher_path(paths: dict, unit: dict) -> Path:
    return teacher_root(paths) / unit["corpus"] / f"{unit['name']}.h5"


def unit_frames(paths: dict, unit: dict) -> list[Path]:
    root = mc.out_root(paths)
    if unit["source"] == "smd":
        f0, f1 = unit["frames"]
        d = root / "frames_cache" / f"smd_{Path(unit['video']).stem}_{f0:06d}_{f1:06d}" / "frames"
        files = sorted(d.glob("*.png"))
    else:
        prefix, first, last = unit["clip"].rsplit(":", 2)
        files = lars_io.clip_frames(paths["lars"]["root"], unit["subset"], prefix, int(first),
                                    int(last))
    if len(files) != unit["n_frames"]:
        raise RuntimeError(f"{unit['key']}: {len(files)} frames, plan says {unit['n_frames']}")
    return files


def unit_manifest(paths: dict, unit: dict) -> dict:
    return json.loads((mc.out_root(paths) / unit["corpus"] / f"{unit['name']}.json").read_text())


def run_unit(paths: dict, unit: dict, teacher: raft.RaftTeacher, cfg_t: dict) -> dict:
    import cv2

    out = teacher_path(paths, unit)
    if out.exists():
        return {"key": unit["key"], "skipped": True}
    files = unit_frames(paths, unit)
    man = unit_manifest(paths, unit)
    sha = sha256_frames(files)
    if sha != man["input"]["sha256"]:
        raise RuntimeError(f"{unit['key']}: frames SHA differs from the v2e unit input")
    w, h = man["conversion"]["width"], man["conversion"]["height"]
    frames = [cv2.cvtColor(resize_like_v2e(cv2.imread(str(f), cv2.IMREAD_COLOR), (w, h)),
                           cv2.COLOR_BGR2RGB) for f in files]
    t0 = time.time()
    res = teacher.pairs(frames, cfg_t["alpha1"], cfg_t["alpha2"])
    wall = time.time() - t0
    write(out, res, {
        "unit": unit["key"], "frames_sha256": sha, "v2e_unit_store_sha256": man["store_sha256"],
        "model": "torchvision raft_large", "weights": teacher.weights,
        "weights_url": teacher.weights_url or "", "iterations": teacher.iters,
        "alpha1": cfg_t["alpha1"], "alpha2": cfg_t["alpha2"],
        "fb_form": "|f + b(x+f)|^2 < alpha1 (|f|^2 + |b(x+f)|^2) + alpha2, x+f inside (D-055)",
        "resize": "cv2 INTER_AREA to the v2e output size, as v2e (D-054)",
        "flow_units": "px per frame interval, forward, frames k -> k+1",
        "wall_time_s": round(wall, 2), "gpu": gpu_name(), "git_sha": git_sha(),
        "git_dirty": git_dirty(), "created_by": "scripts/run_teacher.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return {"key": unit["key"], "n_windows": len(res["flow"]), "wall_time_s": round(wall, 2),
            "coverage": float(res["valid"].mean())}


def summarise(paths: dict, plan: dict, corpora) -> dict:
    rows, missing = [], []
    for u in plan["units"]:
        if u["corpus"] not in corpora:
            continue
        p = teacher_path(paths, u)
        if not p.exists():
            missing.append(u["key"])
            continue
        with TeacherCache(p) as tc:
            valid = tc._f["valid"][:]
            lit = tc._f["valid_literal"][:]
            mag = []
            for k in range(tc.n):
                f, v = tc.get(k)
                m = np.sqrt(f[0] ** 2 + f[1] ** 2)
                mag.append(float(np.median(m[v])) if v.any() else float("nan"))
            row = {"key": u["key"], "corpus": u["corpus"], "n_windows": tc.n,
                   "frames_sha256": tc.attrs["frames_sha256"],
                   "teacher_sha256": _sha(p), "coverage_fb": float(valid.mean()),
                   "coverage_fb_literal": float(lit.mean()),
                   "median_consistent_flow_px": float(np.nanmedian(mag)),
                   "wall_time_s": float(tc.attrs["wall_time_s"])}
            if u["source"] == "lars" and u["corpus"].startswith("lars_eval"):
                per = {}
                for kf, j in zip(u["keyframes"], u["keyframes_local"], strict=True):
                    if j < 1:
                        continue
                    lab = lars_io.semantic_mask(paths["lars"]["root"], u["subset"], kf,
                                                size=(int(tc.attrs["width"]),
                                                      int(tc.attrs["height"])))
                    cm = lars_io.class_masks(lab)
                    _, v = tc.get(j - 1)
                    for c in (*lars_io.CLASSES, "ignore"):
                        n = int(cm[c].sum())
                        per.setdefault(c, [0, 0])
                        per[c][0] += int((v & cm[c]).sum())
                        per[c][1] += n
                row["keyframe_coverage_by_class"] = {
                    c: (a / b if b else None) for c, (a, b) in per.items()}
                row["keyframe_px_by_class"] = {c: b for c, (_, b) in per.items()}
            rows.append(row)
    by: dict = {}
    for r in rows:
        b = by.setdefault(r["corpus"], {"units": 0, "n_windows": 0, "coverage_fb_mean": 0.0,
                                        "wall_time_s": 0.0})
        b["units"] += 1
        b["n_windows"] += r["n_windows"]
        b["coverage_fb_mean"] += r["coverage_fb"] * r["n_windows"]
        b["wall_time_s"] = round(b["wall_time_s"] + r["wall_time_s"], 1)
    for b in by.values():
        b["coverage_fb_mean"] = b["coverage_fb_mean"] / max(b["n_windows"], 1)
    lars_kf: dict = {}
    for r in rows:
        for c, n in r.get("keyframe_px_by_class", {}).items():
            cov = r["keyframe_coverage_by_class"][c]
            a = lars_kf.setdefault(c, [0.0, 0])
            a[0] += (cov or 0.0) * n
            a[1] += n
    return {
        "kind": "teacher_flows", "tier": "T3a",
        "tier_note": "RAFT teacher on maritime source frames; a frame-based estimate, not ground "
                     "truth; coverage = forward-backward-consistent pixel fraction",
        "plan_step": "Phase 7 step 2", "corpora": list(corpora),
        "teacher": {"model": "torchvision raft_large", "weights": raft.WEIGHTS,
                    "iterations": raft.ITERS},
        "n_units_done": len(rows), "missing": missing, "by_corpus": by,
        "lars_eval_keyframe_coverage_by_class": {
            c: (a / n if n else None) for c, (a, n) in lars_kf.items()},
        "units": rows, "git_sha": git_sha(), "git_dirty": git_dirty(),
        "created_by": "scripts/run_teacher.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _sha(p: Path) -> str:
    from evflow.sim.v2e_runner import sha256_file

    return sha256_file(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", action="append", choices=CORPORA)
    ap.add_argument("--limit", type=int, default=None, help="units per corpus (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULT)
    a = ap.parse_args(argv)
    paths = yaml.safe_load((REPO / "configs" / "paths.local.yaml").read_text())
    cfg_t = yaml.safe_load((REPO / "configs" / "evflow.yaml").read_text())["metrics"][
        "teacher"]["fb_consistency"]
    plan = mc.load_plan(paths)
    corpora = tuple(a.corpus or CORPORA)
    if not a.summarise_only:
        teacher = raft.RaftTeacher(device=a.device)
        for c in corpora:
            units = [u for u in plan["units"] if u["corpus"] == c][: a.limit]
            for u in units:
                r = run_unit(paths, u, teacher, cfg_t)
                print(json.dumps(r), flush=True)
    if not a.no_summary:
        doc = summarise(paths, plan, CORPORA)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"units": doc["n_units_done"], "missing": len(doc["missing"]),
                          "by_corpus": doc["by_corpus"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
