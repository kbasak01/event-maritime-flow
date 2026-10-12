"""Maritime v2e corpus: SMD on-board segments and LaRS clips (IMPLEMENTATION_PLAN Phase 7 step 1;
D-026, D-054).

``configs/v2e_corpus_maritime.yaml`` + the split manifests (+ ffprobe of the SMD videos) → an
ordered list of runner units, each one SMD segment or LaRS clip converted at 640×360 with the
``maritime_640x360`` profile (= the calibrated ``mvsec_cal`` pixel model) and one seed:

| corpus           | split      | units                                        | CT multiplier |
|------------------|------------|----------------------------------------------|---------------|
| trial            | smd/lars   | ≤ 5 s SMD + one LaRS clip at the lowest CT   | 1.0 / 0.67    |
| smd_train        | smd_train  | every video, [0, 5.6] s                      | 1.0           |
| smd_test         | smd_test   | every video, [0, min(10 s, last frame)]      | 1.0           |
| lars_train       | lars_train | 110 clips, every k-th                        | 1.0           |
| lars_eval_ct{m}  | lars_eval  | 40 clips, every k-th, m ∈ ct_variants        | m             |

    python -m evflow.sim.maritime_corpus plan
    python -m evflow.sim.maritime_corpus units [--corpus smd_test] [--exclude trial]
    python -m evflow.sim.maritime_corpus unit --key smd_test/smd_MVI_0790_VIS_OB
    python -m evflow.sim.maritime_corpus summarise   # -> results/v2e_maritime_corpus.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evflow.io import lars as lars_io
from evflow.io import smd as smd_io
from evflow.sim.corpus import ct_overrides
from evflow.sim.splits import load_split

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "v2e_corpus_maritime.yaml"
PROFILES = REPO / "configs" / "v2e_profiles.yaml"
PATHS = REPO / "configs" / "paths.local.yaml"
RESULT = REPO / "results" / "v2e_maritime_corpus.json"


# ---------------------------------------------------------------------------------------------
# pure rules (unit-tested)

def ct_tag(m: float) -> str:
    return f"ct{int(round(100 * float(m))):03d}"


def ct_variant_overrides(base_args: dict, m: float) -> dict[str, str]:
    """Multiplier ``m`` on the profile's calibrated pos/neg and sigma thresholds (relative
    mismatch kept constant, D-054); ``m = 1`` is the profile itself (no override)."""
    if abs(float(m) - 1.0) < 1e-12:
        return {}
    pos, neg = float(base_args["--pos_thres"]), float(base_args["--neg_thres"])
    if pos != neg:
        raise ValueError("CT variants assume symmetric thresholds (D-027)")
    return ct_overrides(round(pos * m, 6), round(float(base_args["--sigma_thres"]) * m, 6))


def _unit(key_corpus, name, *, split, source, profile, seed, overrides, ct, video_s,
          n_frames, **extra) -> dict:
    return {"key": f"{key_corpus}/{name}", "corpus": key_corpus, "name": name, "split": split,
            "source": source, "profile": profile, "overrides": overrides, "seed": int(seed),
            "ct_multiplier": float(ct), "n_frames": int(n_frames), "video_s": float(video_s),
            **extra}


def smd_unit(corpus, split, video, probe, start_s, stop_s, *, profile, seed, name=None,
             overrides=None, ct=1.0) -> dict:
    seg = smd_io.segment(start_s, stop_s, probe["fps"], probe["n_frames"])
    stem = Path(video).stem
    return _unit(corpus, name or f"smd_{stem}", split=split, source="smd", profile=profile,
                 seed=seed, overrides=overrides or {}, ct=ct, video_s=seg["video_s"],
                 n_frames=seg["n_frames"], video=f"{stem}.avi", fps=probe["fps"],
                 segment_s=[seg["start_s"], seg["stop_s"]],
                 frames=[seg["first_frame"], seg["last_frame"]], keyframes_local=[])


def lars_unit(corpus, split, subset, clip, fps, *, profile, seed, name=None, overrides=None,
              ct=1.0) -> dict:
    n = int(clip["last"]) - int(clip["first"]) + 1
    return _unit(corpus, name or f"lars_{clip['clip']}", split=split, source="lars",
                 profile=profile, seed=seed, overrides=overrides or {}, ct=ct,
                 video_s=round((n - 1) / fps, 4), n_frames=n, subset=subset,
                 clip=lars_io.clip_spec(clip), fps=float(fps),
                 keyframes=list(clip["keyframes"]),
                 keyframes_local=lars_io.keyframe_indices(clip))


def plan_units(cfg: dict, profiles: dict, splits: dict[str, dict],
               smd_probe: dict[str, dict]) -> list[dict]:
    """Ordered unit list (trials first). ``smd_probe[stem] = {fps, n_frames}``."""
    from evflow.sim.v2e_runner import resolve_profile

    prof, seed = cfg["profile"], int(cfg["seed"])
    base = resolve_profile(profiles, prof)["args"]
    fps_lars = float(profiles["lars"]["assumed_fps"])
    units: list[dict] = []

    tr = cfg["trials"]
    v = splits[tr["smd"]["split"]]["videos"][int(tr["smd"]["video_index"])]
    units.append(smd_unit("trial", tr["smd"]["split"], v, smd_probe[v], tr["smd"]["start_s"],
                          tr["smd"]["stop_s"], profile=prof, seed=seed, name=f"trial_smd_{v}"))
    ls = splits[tr["lars"]["split"]]
    c = lars_io.every_kth(ls["clips"], int(cfg["lars"]["lars_train"]["n_clips"]))[
        int(tr["lars"]["clip_index"])]
    m = float(tr["lars"]["ct"])
    units.append(lars_unit("trial", ls["split"], ls["subset"], c, fps_lars, profile=prof,
                           seed=seed, name=f"trial_lars_{ct_tag(m)}_{c['clip']}",
                           overrides=ct_variant_overrides(base, m), ct=m))

    for corpus, s in cfg["smd"].items():
        sp = splits[s["split"]]
        for v in sp["videos"]:
            units.append(smd_unit(corpus, sp["split"], v, smd_probe[v], s["start_s"],
                                  s["stop_s"], profile=prof, seed=seed))

    lt = cfg["lars"]["lars_train"]
    sp = splits[lt["split"]]
    for c in lars_io.every_kth(sp["clips"], int(lt["n_clips"])):
        units.append(lars_unit("lars_train", sp["split"], sp["subset"], c, fps_lars,
                               profile=prof, seed=seed))

    le = cfg["lars"]["lars_eval"]
    sp = splits[le["split"]]
    chosen = lars_io.every_kth(sp["clips"], int(le["n_clips"]))
    for m in sorted(float(x) for x in cfg["ct_variants"]):
        ov = ct_variant_overrides(base, m)
        for c in chosen:
            units.append(lars_unit(f"lars_eval_{ct_tag(m)}", sp["split"], sp["subset"], c,
                                   fps_lars, profile=prof, seed=seed, overrides=ov, ct=m))

    keys = [u["key"] for u in units]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate unit keys")
    for u in units:
        if u["video_s"] > 60.0:            # v2e float32 time precision (D-022)
            raise ValueError(f"{u['key']} spans {u['video_s']} s > 60 s")
    return units


# ---------------------------------------------------------------------------------------------
# I/O

def _yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def _paths() -> dict:
    return _yaml(PATHS)


def out_root(paths: dict, cfg: dict | None = None) -> Path:
    cfg = cfg or _yaml(CONFIG)
    return Path(paths["outputs"]["events"]) / cfg["out_root"]


def store_path(paths: dict, unit: dict) -> Path:
    return out_root(paths) / unit["corpus"] / f"{unit['name']}.h5"


def build_plan() -> dict:
    cfg, prof, paths = _yaml(CONFIG), _yaml(PROFILES), _paths()
    names = {s["split"] for s in cfg["smd"].values()} | {
        s["split"] for s in cfg["lars"].values()} | {t["split"] for t in cfg["trials"].values()}
    splits = {n: load_split(n) for n in sorted(names)}
    vids = sorted({v for n in names if n.startswith("smd") for v in splits[n]["videos"]})
    probe = {v: smd_io.probe(smd_io.video_path(paths["smd"]["onboard"], v)) for v in vids}
    units = plan_units(cfg, prof, splits, probe)
    return {"kind": "v2e_corpus_plan", "config": "configs/v2e_corpus_maritime.yaml",
            "split_manifest_sha256": {n: splits[n]["manifest_sha256"] for n in sorted(splits)},
            "calibration_sha256": prof["calibration_sha256"],
            "lars_assumed_fps": float(prof["lars"]["assumed_fps"]), "units": units}


def _plan_file(paths: dict) -> Path:
    return out_root(paths) / "corpus_plan.json"


def load_plan(paths: dict | None = None) -> dict:
    return json.loads(_plan_file(paths or _paths()).read_text())


def cmd_plan() -> dict:
    paths = _paths()
    doc = build_plan()
    f = _plan_file(paths)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def unit_argv(u: dict, root: Path) -> list[str]:
    common = ["--profile", u["profile"], "--unit", u["name"], "--out", str(root / u["corpus"]),
              "--seed", str(u["seed"])]
    if u["source"] == "smd":
        argv = ["smd", "--video", u["video"], "--start", repr(u["segment_s"][0]),
                "--stop", repr(u["segment_s"][1]), "--frames-cache", str(root / "frames_cache")]
    else:
        argv = ["lars", "--subset", u["subset"], "--clip", u["clip"]]
    return argv + common + [f"--set={k}={v}" for k, v in u["overrides"].items()]


def cmd_unit(key: str) -> None:
    paths = _paths()
    plan = load_plan(paths)
    u = next((u for u in plan["units"] if u["key"] == key), None)
    if u is None:
        raise SystemExit(f"unknown unit {key!r}")
    subprocess.run([sys.executable, "-m", "evflow.sim.v2e_runner", *unit_argv(u, out_root(paths))],
                   check=True)


def _unit_row(u: dict, m: dict) -> dict:
    conv = m["conversion"]
    return {**u, "store_sha256": m["store_sha256"],
            "run_manifest_sha256": m["run_manifest_sha256"], "store_bytes": m["store_bytes"],
            "n_events": conv["n_events"], "events_per_s": conv["events_per_s"],
            "on_fraction": conv["on_fraction"], "duration_us": conv["duration_us"],
            "t_offset_us": conv["t_offset_us"], "input_frame_rate": m["input_frame_rate"],
            "frame_rate_source": m["frame_rate_source"], "input_sha256": m["input"]["sha256"],
            "native_size": m["input"].get("native_size"),
            "scale_xy": m["input"].get("scale_xy"), "wall_time_s": m["wall_time_s"],
            "warnings": m["warnings"], "allow_undersampling": m["allow_undersampling"],
            "v2e_commit": m["v2e_commit"], "git_sha": m["git_sha"], "git_dirty": m["git_dirty"],
            "gpu": m["gpu"], "created_at": m["created_at"]}


def cmd_summarise(out: Path = RESULT) -> dict:
    paths = _paths()
    plan = load_plan(paths)
    fresh = build_plan()
    if fresh["units"] != plan["units"]:
        raise SystemExit("corpus_plan.json differs from the plan rebuilt from configs + splits")
    root = out_root(paths)
    rows, missing = [], []
    for u in plan["units"]:
        f = root / u["corpus"] / f"{u['name']}.json"
        if not f.exists():
            missing.append(u["key"])
            continue
        rows.append(_unit_row(u, json.loads(f.read_text())))
    starts = [datetime.fromisoformat(r["created_at"]).timestamp() - r["wall_time_s"]
              for r in rows]
    ends = [datetime.fromisoformat(r["created_at"]).timestamp() for r in rows]
    by: dict[str, dict] = {}
    for r in rows:
        b = by.setdefault(r["corpus"], {"units": 0, "video_s": 0.0, "sum_unit_wall_s": 0.0,
                                        "n_events": 0})
        b["units"] += 1
        b["video_s"] = round(b["video_s"] + r["video_s"], 4)
        b["sum_unit_wall_s"] = round(b["sum_unit_wall_s"] + r["wall_time_s"], 2)
        b["n_events"] += r["n_events"]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    doc = {
        "kind": "v2e_corpus", "tier": "n/a",
        "tier_note": "v2e(SMD, LaRS) maritime event corpus (synthetic, used for T3 rows); "
                     "no metrics",
        "plan_step": "Phase 7 step 1", "config": plan["config"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "lars_assumed_fps": plan["lars_assumed_fps"],
        "n_units_planned": len(plan["units"]), "n_units_done": len(rows), "missing": missing,
        "checks": {
            "all_git_clean": all(r["git_dirty"] is False for r in rows),
            "undersampling_warnings": sum(len(r["warnings"]["undersampling"]) for r in rows),
            "too_many_events_warnings": sum(int(r["warnings"].get("too_many_events", 0))
                                            for r in rows),
            "any_allow_undersampling": any(r["allow_undersampling"] for r in rows),
            "git_shas": sorted({r["git_sha"] for r in rows}),
        },
        "by_corpus": by,
        "totals": {
            "video_s": round(sum(r["video_s"] for r in rows), 2),
            "sum_unit_wall_s": round(sum(r["wall_time_s"] for r in rows), 1),
            "job_span_h": round((max(ends) - min(starts)) / 3600, 3) if rows else None,
            "store_bytes": sum(r["store_bytes"] for r in rows),
        },
        "units": rows,
        "git_sha": sha, "created_by": "src/evflow/sim/maritime_corpus.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    u = sub.add_parser("units")
    u.add_argument("--corpus", action="append", help="only these corpora (repeatable)")
    u.add_argument("--exclude", action="append", default=[], help="skip these corpora")
    one = sub.add_parser("unit")
    one.add_argument("--key", required=True, help="<corpus>/<name>")
    s = sub.add_parser("summarise")
    s.add_argument("--out", type=Path, default=RESULT)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        d = cmd_plan()
        by: dict = {}
        for x in d["units"]:
            b = by.setdefault(x["corpus"], [0, 0.0])
            b[0] += 1
            b[1] += x["video_s"]
        print(json.dumps({k: {"units": n, "video_s": round(v, 2)} for k, (n, v) in by.items()}))
    elif a.cmd == "units":
        for x in load_plan()["units"]:
            if (a.corpus is None or x["corpus"] in a.corpus) and x["corpus"] not in a.exclude:
                print(x["key"])
    elif a.cmd == "unit":
        cmd_unit(a.key)
    else:
        d = cmd_summarise(a.out)
        print(json.dumps({"done": d["n_units_done"], "missing": d["missing"], **d["checks"],
                          **d["totals"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
