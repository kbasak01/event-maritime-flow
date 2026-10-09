"""Matched-content MVSEC v2e(APS) corpus (IMPLEMENTATION_PLAN Phase 3 step 1; D-026, D-032).

``configs/v2e_corpus_mvsec.yaml`` + the split manifests + the APS clocks → an ordered list of
runner units, each one MVSEC APS frame range converted with one profile and seed:

| corpus      | split       | units                                              | profile      |
|-------------|-------------|----------------------------------------------------|--------------|
| trial       | mvsec_tune  | 2 × ~5 s at the CT-randomisation extremes          | mvsec_rand   |
| tune        | mvsec_tune  | the whole 60 s slice                               | mvsec_cal    |
| test        | mvsec_test  | every window of every sequence, ≤ 60 s chunks       | mvsec_cal    |
| train_cal   | mvsec_train | 4 × 60 s clips (D-026)                             | mvsec_cal    |
| rand{k}     | mvsec_train | the same clips, CT drawn per clip, k = 0..K-1      | mvsec_rand   |

A unit *owns* a run of APS window starts. Test chunks overlap: a chunk starts ``warmup_frames``
before its first owned window (v2e state settles; before the first window of a sequence only via
the runner's lead-frame rule) and ends ``max_dt`` frames past its last owned start, so every dt=1
and dt=4 window of the test split is owned by exactly one unit. Statistics use owned windows
from ``first + warmup_frames`` on (``stats_from``).

    python -m evflow.sim.corpus plan        # -> <events>/v2e/mvsec/corpus_plan.json
    python -m evflow.sim.corpus prepare     # export every APS range once (before parallel units)
    python -m evflow.sim.corpus units [--corpus trial]   # unit names, for xargs
    python -m evflow.sim.corpus unit --key test/test_indoor_flying1_p0
    python -m evflow.sim.corpus summarise   # -> results/v2e_mvsec_corpus.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from evflow.sim.splits import MAX_LEAD_FRAMES, check_aps_in_split, load_split

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "v2e_corpus_mvsec.yaml"
PROFILES = REPO / "configs" / "v2e_profiles.yaml"
PATHS = REPO / "configs" / "paths.local.yaml"
RESULT = REPO / "results" / "v2e_mvsec_corpus.json"
TRAIN_SEQ = "outdoor_day2"


# ---------------------------------------------------------------------------------------------
# pure rules (unit-tested)

def frames_in(aps_us: np.ndarray, t0: int, t1: int) -> tuple[int, int]:
    """Inclusive APS frame range with timestamps in [t0, t1)."""
    i0 = int(np.searchsorted(aps_us, t0, side="left"))
    i1 = int(np.searchsorted(aps_us, t1, side="left")) - 1
    return i0, i1


def train_clips(split: dict, aps_us: np.ndarray, n_clips: int, clip_s: float,
                range_index: int) -> list[tuple[int, int]]:
    """D-026: starts = int(round(t)) for t in linspace(r0, r1 - clip, n) on ranges_us[i];
    a clip holds the frames with t in [start, start + clip)."""
    r0, r1 = (int(v) for v in split["ranges_us"][range_index])
    clip_us = int(round(clip_s * 1e6))
    starts = [int(round(t)) for t in np.linspace(r0, r1 - clip_us, n_clips)]
    return [frames_in(aps_us, s, s + clip_us) for s in starts]


def chunk_test_sequence(seq: dict, aps_us: np.ndarray, warmup: int, max_unit_s: float,
                max_dt: int) -> list[dict]:
    """Fewest equal chunks of the window starts such that every unit (lead-in + owned starts +
    ``max_dt`` tail) spans at most ``max_unit_s``."""
    w = seq["windows"]
    first_start = min(int(v["aps_first"]) for v in w.values())
    last_start = max(int(v["aps_last"]) for v in w.values())
    last_frame = max(int(v["aps_last"]) + int(k.removeprefix("dt")) for k, v in w.items())
    n_starts = last_start - first_start + 1
    for n in range(1, n_starts + 1):
        b = [first_start + int(round(j * n_starts / n)) for j in range(n + 1)]
        units = []
        for j in range(n):
            first = max(b[j] - warmup, 0)
            last = min(b[j + 1] - 1 + max_dt, last_frame)
            units.append({"aps": [first, last], "owned": [b[j], b[j + 1] - 1],
                          "lead_frames": max(first_start - first, 0)})
        if all(aps_us[u["aps"][1]] - aps_us[u["aps"][0]] <= max_unit_s * 1e6 for u in units):
            return units
    raise ValueError(f"{seq['sequence']}: cannot chunk to <= {max_unit_s} s")


def rand_draws(randomise: dict, draw_seed: int, n_clips: int) -> list[dict]:
    """pos_thres = neg_thres log-uniform in thres_range, sigma_thres uniform in sigma_range, per
    (corpus, clip) in that order (plan Phase 3 step 1; ranges frozen in Phase 2)."""
    lo, hi = (float(v) for v in randomise["thres_range"])
    slo, shi = (float(v) for v in randomise["sigma_range"])
    rng = np.random.default_rng(draw_seed)
    out = []
    for k in range(int(randomise["corpora"])):
        for c in range(n_clips):
            th = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            sg = float(rng.uniform(slo, shi))
            out.append({"corpus": k, "clip": c, "pos_neg_thres": round(th, 6),
                        "sigma_thres": round(sg, 6)})
    return out


def ct_overrides(thres: float, sigma: float) -> dict[str, str]:
    return {"--pos_thres": repr(float(thres)), "--neg_thres": repr(float(thres)),
            "--sigma_thres": repr(float(sigma))}


def _unit(name, corpus, split, sequence, aps, owned, lead, profile, seed, warmup,
          overrides=None, **extra) -> dict:
    return {"name": name, "corpus": corpus, "split": split, "sequence": sequence,
            "aps": [int(aps[0]), int(aps[1])], "owned_window_starts": [int(owned[0]),
                                                                       int(owned[1])],
            "stats_from_window": int(aps[0]) + warmup, "lead_frames": int(lead),
            "profile": profile, "overrides": overrides or {}, "seed": int(seed), **extra}


def plan_units(cfg: dict, profiles: dict, splits: dict[str, dict],
               aps: dict[str, np.ndarray]) -> list[dict]:
    """Ordered unit list: trials, tune, test, train_cal, rand0..K-1. Every range is checked
    against its split (LeakageError) before it is returned."""
    W, max_dt = int(cfg["warmup_frames"]), int(cfg["max_dt"])
    if W > MAX_LEAD_FRAMES:
        raise ValueError("warmup_frames exceeds the runner's lead-frame limit")
    cal_prof, cal_seed = cfg["cal"]["profile"], int(cfg["cal"]["seed"])
    rnd = profiles["profiles"][cfg["rand"]["profile"]]["randomise"]
    units: list[dict] = []

    # trials at the randomisation extremes, from the start of the tune slice
    tr = cfg["trials"]
    t_split = splits[tr["split"]]
    f0 = int(t_split["aps"][0]["aps_first"])
    ext = {"lo": (rnd["thres_range"][0], rnd["sigma_range"][1]),
           "hi": (rnd["thres_range"][1], rnd["sigma_range"][0])}
    for e in tr["extremes"]:
        th, sg = ext[e]
        a = (f0, f0 + int(tr["n_frames"]) - 1)
        units.append(_unit(f"trial_rand_{e}", "trial", tr["split"], TRAIN_SEQ, a,
                           (a[0], a[1] - 1), 0, cfg["rand"]["profile"], cal_seed, W,
                           ct_overrides(th, sg), pos_neg_thres=float(th),
                           sigma_thres=float(sg)))

    # tune: the whole slice
    tune = splits[cfg["tune"]["split"]]
    for r in tune["aps"]:
        a = (int(r["aps_first"]), int(r["aps_last"]))
        units.append(_unit(f"tune_{TRAIN_SEQ}", "tune", tune["split"], TRAIN_SEQ, a,
                           (a[0], a[1] - 1), 0, cal_prof, cal_seed, W))

    # test: every window, chunked
    test = splits[cfg["test"]["split"]]
    for seq in test["sequences"]:
        s = seq["sequence"]
        chunks = chunk_test_sequence(seq, aps[s], W, float(cfg["max_unit_s"]), max_dt)
        for j, c in enumerate(chunks):
            units.append(_unit(f"test_{s}_p{j}", "test", test["split"], s, c["aps"],
                               c["owned"], c["lead_frames"], cal_prof, cal_seed, W))

    # train: D-026 clips, calibrated then randomised
    trc = cfg["train"]
    train = splits[trc["split"]]
    clips = train_clips(train, aps[TRAIN_SEQ], int(trc["n_clips"]), float(trc["clip_s"]),
                        int(trc["range_index"]))
    for c, a in enumerate(clips):
        units.append(_unit(f"train_clip{c}", "train_cal", train["split"], TRAIN_SEQ, a,
                           (a[0], a[1] - 1), 0, cal_prof, cal_seed, W, clip=c))
    for d in rand_draws(rnd, int(cfg["rand"]["draw_seed"]), len(clips)):
        k, c = d["corpus"], d["clip"]
        a = clips[c]
        units.append(_unit(f"train_clip{c}", f"rand{k}", train["split"], TRAIN_SEQ, a,
                           (a[0], a[1] - 1), 0, cfg["rand"]["profile"],
                           int(cfg["rand"]["seed_base"]) + 100 * k + c, W,
                           ct_overrides(d["pos_neg_thres"], d["sigma_thres"]), clip=c,
                           pos_neg_thres=d["pos_neg_thres"], sigma_thres=d["sigma_thres"]))

    for u in units:
        check_aps_in_split(splits[u["split"]], u["sequence"], u["aps"][0], u["aps"][1],
                           u["lead_frames"])
        u["key"] = f"{u['corpus']}/{u['name']}"
        u["video_s"] = round(float(aps[u["sequence"]][u["aps"][1]]
                                   - aps[u["sequence"]][u["aps"][0]]) / 1e6, 4)
        if u["video_s"] > float(cfg["max_unit_s"]):
            raise ValueError(f"{u['key']} spans {u['video_s']} s > max_unit_s")
    keys = [u["key"] for u in units]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate unit keys")
    return units


# ---------------------------------------------------------------------------------------------
# I/O

def _yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def _paths() -> dict:
    return _yaml(PATHS)


def _data_file(paths: dict, sequence: str) -> Path:
    scene = re.sub(r"\d+$", "", sequence)
    return Path(paths["mvsec"]["root"]) / scene / f"{sequence}_data.hdf5"


def out_root(paths: dict, cfg: dict | None = None) -> Path:
    cfg = cfg or _yaml(CONFIG)
    return Path(paths["outputs"]["events"]) / cfg["out_root"]


def build_plan() -> dict:
    from evflow.io.mvsec import read_aps_timestamps_us

    cfg, prof, paths = _yaml(CONFIG), _yaml(PROFILES), _paths()
    names = {cfg[k]["split"] for k in ("tune", "test", "train", "trials")}
    splits = {n: load_split(n) for n in names}
    seqs = {TRAIN_SEQ} | {s["sequence"] for s in splits[cfg["test"]["split"]]["sequences"]}
    aps = {s: read_aps_timestamps_us(_data_file(paths, s)) for s in sorted(seqs)}
    units = plan_units(cfg, prof, splits, aps)
    return {"kind": "v2e_corpus_plan", "config": "configs/v2e_corpus_mvsec.yaml",
            "split_manifest_sha256": {n: splits[n]["manifest_sha256"] for n in sorted(splits)},
            "calibration_sha256": prof["calibration_sha256"], "units": units}


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


def cmd_prepare() -> int:
    from evflow.sim.v2e_runner import export_mvsec_aps, mvsec_cache_dir

    paths = _paths()
    plan = load_plan(paths)
    cache = out_root(paths) / "aps_cache"
    done = set()
    for u in plan["units"]:
        k = (u["sequence"], *u["aps"])
        if k in done:
            continue
        export_mvsec_aps(_data_file(paths, u["sequence"]),
                         mvsec_cache_dir(cache, u["sequence"], *u["aps"]), *u["aps"])
        done.add(k)
    return len(done)


def cmd_unit(key: str) -> None:
    paths = _paths()
    plan = load_plan(paths)
    u = next((u for u in plan["units"] if u["key"] == key), None)
    if u is None:
        raise SystemExit(f"unknown unit {key!r}")
    root = out_root(paths)
    argv = ["mvsec", "--sequence", u["sequence"], "--aps", str(u["aps"][0]), str(u["aps"][1]),
            "--split", u["split"], "--lead-frames", str(u["lead_frames"]),
            "--frames-cache", str(root / "aps_cache"), "--profile", u["profile"],
            "--unit", u["name"], "--out", str(root / u["corpus"]), "--seed", str(u["seed"])]
    argv += [f"--set={k}={v}" for k, v in u["overrides"].items()]
    subprocess.run([sys.executable, "-m", "evflow.sim.v2e_runner", *argv], check=True)


def _unit_row(u: dict, m: dict) -> dict:
    conv = m["conversion"]
    return {**{k: u[k] for k in ("key", "corpus", "name", "split", "sequence", "aps",
                                 "owned_window_starts", "stats_from_window", "lead_frames",
                                 "profile", "overrides", "seed", "video_s")},
            "store_sha256": m["store_sha256"], "run_manifest_sha256": m["run_manifest_sha256"],
            "store_bytes": m["store_bytes"], "n_events": conv["n_events"],
            "events_per_s": conv["events_per_s"], "on_fraction": conv["on_fraction"],
            "n_events_beyond_clock_map": conv.get("n_events_beyond_clock_map"),
            "clock_map": m.get("clock_map"), "input_frame_rate": m["input_frame_rate"],
            "input_sha256": m["input"]["sha256"], "wall_time_s": m["wall_time_s"],
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
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    doc = {
        "kind": "v2e_corpus", "tier": "T2",
        "tier_note": "v2e(MVSEC APS) event corpus for the matched-content arms; no metrics",
        "plan_step": "Phase 3 step 1", "config": plan["config"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "n_units_planned": len(plan["units"]), "n_units_done": len(rows), "missing": missing,
        "checks": {
            "all_git_clean": all(r["git_dirty"] is False for r in rows),
            "undersampling_warnings": sum(len(r["warnings"]["undersampling"]) for r in rows),
            "any_allow_undersampling": any(r["allow_undersampling"] for r in rows),
            "max_clamped_fraction": max((r["n_events_beyond_clock_map"] or 0) / max(
                r["n_events"], 1) for r in rows) if rows else None,
            "git_shas": sorted({r["git_sha"] for r in rows}),
        },
        "totals": {
            "video_s": round(sum(r["video_s"] for r in rows), 2),
            "sum_unit_wall_s": round(sum(r["wall_time_s"] for r in rows), 1),
            "job_span_h": round((max(ends) - min(starts)) / 3600, 3) if rows else None,
            "store_bytes": sum(r["store_bytes"] for r in rows),
        },
        "units": rows,
        "git_sha": sha, "created_by": "src/evflow/sim/corpus.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("prepare")
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
        by = {}
        for x in d["units"]:
            by.setdefault(x["corpus"], [0, 0.0])
            by[x["corpus"]][0] += 1
            by[x["corpus"]][1] += x["video_s"]
        print(json.dumps({k: {"units": n, "video_s": round(v, 2)} for k, (n, v) in by.items()}))
    elif a.cmd == "prepare":
        print(json.dumps({"aps_ranges_exported": cmd_prepare()}))
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
