"""Write the fixed data-split manifests to ``splits/`` (plan §1.3, Phase 2 step 0).

    python scripts/make_splits.py [--check]

Seed 20261008 (configs/evflow.yaml ``split_seed``). Every manifest records its rule, the source
files (size + SHA-256; large files hashed once and cached in ``data/_sha256_cache.json``), and its
own content SHA-256 (``manifest_sha256``, over the canonical JSON without that key). ``--check``
regenerates in memory and fails if any committed manifest differs.

MVSEC times are absolute µs on the MVSEC clock (``round(unix_s * 1e6)``), the clock of
``EventSlicer`` on converted MVSEC stores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from check_data import git_state, redact_home, sha256_file

from evflow.io.mvsec import GTFlow, read_aps_timestamps_us

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "splits"
SHA_CACHE = REPO / "data" / "_sha256_cache.json"

TUNE_LEN_US = 60_000_000          # §1.3: one contiguous 60 s slice
TUNE_GAP_US = 30_000_000          # §1.3: >= 30 s gap from train windows
TUNE_GRID_US = 1_000_000          # start drawn on a 1 s grid
STATIC_MEDIAN_PX = 0.1            # tune slice rejected if median per-field median |GT| < this
SMD_N_TEST = 3                    # §1.3: SMD split by video 8 / 3
MVSEC_TEST = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
DSEC_TEST = ["thun_00_a", "zurich_city_02_a", "zurich_city_11_a"]   # D-012


# ---------------------------------------------------------------------------------------------
# helpers

def _load_cache() -> dict:
    try:
        return json.loads(SHA_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def file_entry(path: Path, cache: dict) -> dict:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if key not in cache:
        cache[key] = sha256_file(path)
    return {"path": redact_home(path), "size_bytes": st.st_size, "sha256": cache[key]}


def listing_sha256(files: list[Path]) -> str:
    """SHA-256 of the sorted (name, size) listing: identifies a large image folder cheaply."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.name):
        h.update(f"{f.name}\t{f.stat().st_size}\n".encode())
    return h.hexdigest()


def finalize(doc: dict) -> dict:
    body = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return {**doc, "manifest_sha256": hashlib.sha256(body).hexdigest()}


# ---------------------------------------------------------------------------------------------
# MVSEC

def draw_tune_slice(gt_ts_us: np.ndarray, seed: int, is_static) -> tuple[int, list[dict]]:
    """Seeded draw of the tune start on a 1 s grid inside the GT span; redraw while static.

    ``is_static(t0, t1) -> (bool, float)``. Returns the start (abs µs) and the draw log."""
    g0, g1 = int(gt_ts_us[0]), int(gt_ts_us[-1])
    n_cand = (g1 - g0 - TUNE_LEN_US) // TUNE_GRID_US + 1
    if n_cand <= 0:
        raise ValueError("GT span shorter than the tune slice")
    rng = np.random.default_rng(seed)
    log = []
    for _ in range(100):
        k = int(rng.integers(n_cand))
        t0 = g0 + k * TUNE_GRID_US
        static, med = is_static(t0, t0 + TUNE_LEN_US)
        log.append({"grid_index": k, "t_start_us": t0, "median_gt_px": round(med, 4),
                    "rejected_static": bool(static)})
        if not static:
            return t0, log
    raise RuntimeError("no non-static tune slice in 100 draws")


def train_ranges(g0: int, g1: int, tune0: int) -> list[list[int]]:
    """GT span minus [tune0 - gap, tune1 + gap]; half-open [a, b) ranges, empty ones dropped."""
    tune1 = tune0 + TUNE_LEN_US
    out = []
    if tune0 - TUNE_GAP_US > g0:
        out.append([g0, tune0 - TUNE_GAP_US])
    if tune1 + TUNE_GAP_US < g1:
        out.append([tune1 + TUNE_GAP_US, g1])
    return out


def aps_frames_in(aps_us: np.ndarray, a: int, b: int) -> dict:
    """APS frame indices with timestamps in [a, b) and dt=1 window count inside the range."""
    i0 = int(np.searchsorted(aps_us, a, side="left"))
    i1 = int(np.searchsorted(aps_us, b, side="left"))   # exclusive
    return {"aps_first": i0, "aps_last": i1 - 1, "n_frames": i1 - i0,
            "n_windows_dt1": max(i1 - i0 - 1, 0), "n_windows_dt4": max(i1 - i0 - 4, 0)}


def mvsec_outdoor_day2(cfg: dict, seed: int, cache: dict) -> tuple[dict, dict]:
    root, gtroot = Path(cfg["mvsec"]["root"]), Path(cfg["mvsec"]["gt_flow"])
    data = root / "outdoor_day" / "outdoor_day2_data.hdf5"
    gtp = gtroot / "outdoor_day" / "outdoor_day2_gt_flow_dist.npz"
    gt = GTFlow(gtp)
    aps = read_aps_timestamps_us(data)
    ts = gt.timestamps_us

    def is_static(a: int, b: int) -> tuple[bool, float]:
        idx = np.nonzero((ts >= a) & (ts < b))[0][::20]          # one field per second (20 Hz)
        meds = []
        for i in idx:
            f = gt.field(int(i))
            mag = np.hypot(f[0], f[1])
            meds.append(float(np.median(mag[np.isfinite(mag)])))
        med = float(np.median(meds))
        return med < STATIC_MEDIAN_PX, med

    tune0, log = draw_tune_slice(ts, seed, is_static)
    g0, g1 = int(ts[0]), int(ts[-1])
    tune = [tune0, tune0 + TUNE_LEN_US]
    tr = train_ranges(g0, g1, tune0)
    sources = [file_entry(data, cache), file_entry(gtp, cache)]
    common = {"dataset": "mvsec", "sequence": "outdoor_day2", "camera": "left",
              "clock": "absolute us, round(unix_s*1e6)", "sources": sources,
              "gt_span_us": [g0, g1], "seed": seed}
    t_rel = lambda r: [round((r[0] - int(aps[0])) / 1e6, 3), round((r[1] - int(aps[0])) / 1e6, 3)]  # noqa: E731
    tune_doc = {
        "split": "mvsec_tune", **common,
        "rule": (f"one contiguous {TUNE_LEN_US // 10**6} s slice, start drawn uniformly on a "
                 f"{TUNE_GRID_US // 10**6} s grid inside the GT span with "
                 f"numpy default_rng({seed}); redrawn while the median over 1 Hz-sampled GT fields "
                 f"of the per-field median |GT| < {STATIC_MEDIAN_PX} px"),
        "ranges_us": [tune], "ranges_s_from_first_aps": [t_rel(tune)],
        "aps": [aps_frames_in(aps, *tune)], "draws": log,
        "used_for": ["v2e calibration", "model/epoch selection", "CT-randomisation range"],
    }
    train_doc = {
        "split": "mvsec_train", **common,
        "rule": (f"GT span minus [tune_start - {TUNE_GAP_US // 10**6} s, tune_end + "
                 f"{TUNE_GAP_US // 10**6} s); windows must lie fully inside one range"),
        "ranges_us": tr, "ranges_s_from_first_aps": [t_rel(r) for r in tr],
        "aps": [aps_frames_in(aps, *r) for r in tr],
        "excluded_gap_us": TUNE_GAP_US, "used_for": ["training every MVSEC arm"],
    }
    return train_doc, tune_doc


def mvsec_test(cfg: dict, cache: dict) -> dict:
    ev = yaml.safe_load((REPO / "configs/eval_mvsec.yaml").read_text())
    root, gtroot = Path(cfg["mvsec"]["root"]), Path(cfg["mvsec"]["gt_flow"])
    seqs = []
    for seq in MVSEC_TEST:
        scene = re.sub(r"\d+$", "", seq)
        data = root / scene / f"{seq}_data.hdf5"
        gtp = gtroot / scene / f"{seq}_gt_flow_dist.npz"
        aps = read_aps_timestamps_us(data)
        with np.load(gtp, mmap_mode="r") as z:
            gts = np.round(np.asarray(z["timestamps"], np.float64) * 1e6).astype(np.int64)
        rule = ev["sequences"][seq]["windows"]
        if rule["rule"] == "all_in_gt_span":
            a, b = int(gts[0]), int(gts[-1])
        else:
            a = int(round(rule["t_start_s"] * 1e6))
            b = int(round(rule["t_end_s"] * 1e6))
        win = {}
        for dt in ev["dt_frames"]:
            k = np.nonzero((aps[:-dt] >= a) & (aps[dt:] <= b))[0]
            win[f"dt{dt}"] = {"aps_first": int(k[0]), "aps_last": int(k[-1]), "n_windows": len(k)}
        seqs.append({"sequence": seq, "camera": "left", "rule": rule["rule"],
                     "range_us": [a, b], "windows": win,
                     "sources": [file_entry(data, cache), file_entry(gtp, cache)]})
    return {"split": "mvsec_test", "dataset": "mvsec", "protocol": "configs/eval_mvsec.yaml",
            "clock": "absolute us, round(unix_s*1e6)", "sequences": seqs,
            "used_for": ["T1/T2 evaluation only"]}


# ---------------------------------------------------------------------------------------------
# DSEC / SMD / LaRS

def dsec_test(cfg: dict, cache: dict) -> dict:
    root = Path(cfg["dsec"]["root"])
    seqs = []
    for seq in DSEC_TEST:
        d = root / seq
        small = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in {".txt", ".yaml",
                                                                                ".csv"})
        big = [d / "events_left" / "events.h5", d / "events_left" / "rectify_map.h5"]
        pngs = sorted((d / "flow").rglob("*.png"))
        seqs.append({"sequence": seq, "files": [file_entry(p, cache) for p in small + big],
                     "flow_png_count": len(pngs), "flow_png_listing_sha256": listing_sha256(pngs)})
    return {"split": "dsec_test", "dataset": "dsec", "decision": "D-012", "sequences": seqs,
            "used_for": ["T1 generalisation evaluation only"]}


def smd(cfg: dict, seed: int, cache: dict) -> tuple[dict, dict]:
    vids = sorted((Path(cfg["smd"]["onboard"]) / "Videos").glob("*.avi"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vids))
    test = sorted(vids[i] for i in perm[:SMD_N_TEST])
    train = sorted(vids[i] for i in perm[SMD_N_TEST:])
    rule = (f"by video: numpy default_rng({seed}).permutation over the sorted VIS_Onboard AVIs; "
            f"first {SMD_N_TEST} -> smd_test, rest -> smd_train")
    mk = lambda name, vs, use: {"split": name, "dataset": "smd", "subset": "VIS_Onboard",  # noqa: E731
                                "rule": rule, "seed": seed, "videos": [v.stem for v in vs],
                                "files": [file_entry(v, cache) for v in vs], "used_for": use}
    return (mk("smd_train", train, ["maritime training"]),
            mk("smd_test", test, ["T3 evaluation only"]))


def lars_clips(split_dir: Path) -> list[dict]:
    """Clips = maximal runs of consecutive frame numbers (same prefix) in ``images_seq``; each run
    holds the 9 frames before its keyframe(s) in ``images`` and up to 5 after."""
    keys = {p.stem for p in (split_dir / "images").iterdir()}
    frames = sorted(p.stem for p in (split_dir / "images_seq").iterdir())
    runs: list[list[tuple[str, int, str]]] = []
    for s in frames:
        pre, num = s.rsplit("_", 1)
        n = int(num)
        if runs and runs[-1][-1][0] == pre and runs[-1][-1][1] == n - 1:
            runs[-1].append((pre, n, s))
        else:
            runs.append([(pre, n, s)])
    out = []
    for r in runs:
        kf = [s for _, _, s in r if s in keys]
        out.append({"clip": r[0][2], "prefix": r[0][0], "first": r[0][1], "last": r[-1][1],
                    "n_frames": len(r), "keyframes": kf})
    return out


def lars(cfg: dict) -> tuple[dict, dict]:
    root = Path(cfg["lars"]["root"])
    docs = []
    for name, sub, use in [("lars_train", "train", ["maritime training"]),
                           ("lars_eval", "val", ["class-sliced T3 evaluation only"])]:
        d = root / sub
        clips = lars_clips(d)
        seq_files = sorted((d / "images_seq").iterdir())
        docs.append({"split": name, "dataset": "lars", "subset": sub,
                     "rule": "all clips of the LaRS split (test annotations are not public)",
                     "n_clips": len(clips), "n_frames": len(seq_files),
                     "n_clips_without_keyframe": sum(not c["keyframes"] for c in clips),
                     "images_seq_listing_sha256": listing_sha256(seq_files),
                     "clips": clips, "used_for": use})
    return docs[0], docs[1]


# ---------------------------------------------------------------------------------------------

def build(cfg: dict, seed: int, cache: dict) -> dict[str, dict]:
    train, tune = mvsec_outdoor_day2(cfg, seed, cache)
    smd_tr, smd_te = smd(cfg, seed, cache)
    lars_tr, lars_ev = lars(cfg)
    docs = [train, tune, mvsec_test(cfg, cache), dsec_test(cfg, cache), smd_tr, smd_te,
            lars_tr, lars_ev]
    return {d["split"]: finalize({**d, "created_by": "scripts/make_splits.py"}) for d in docs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=Path, default=REPO / "configs/paths.local.yaml")
    ap.add_argument("--check", action="store_true", help="fail if splits/ differs")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.paths.read_text())
    seed = int(yaml.safe_load((REPO / "configs/evflow.yaml").read_text())["split_seed"])
    cache = _load_cache()
    docs = build(cfg, seed, cache)
    SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SHA_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    os.replace(tmp, SHA_CACHE)

    if args.check:
        bad = [n for n, d in docs.items()
               if not (OUT / f"{n}.json").exists()
               or (OUT / f"{n}.json").read_bytes() != (json.dumps(d, indent=1) + "\n").encode()]
        sums = "".join(f"{d['manifest_sha256']}  {n}.json\n" for n, d in docs.items())
        if not (OUT / "SHA256SUMS.txt").exists() or (OUT / "SHA256SUMS.txt").read_text() != sums:
            bad.append("SHA256SUMS.txt")
        print("splits OK" if not bad else f"splits differ: {bad}")
        return 1 if bad else 0

    OUT.mkdir(exist_ok=True)
    sha, dirty = git_state()
    lines = []
    for name, d in docs.items():
        (OUT / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
        lines.append(f"{d['manifest_sha256']}  {name}.json")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    (OUT / "README.md").write_text(
        "# splits\n\nGenerated by `scripts/make_splits.py` (never edit by hand). "
        f"Seed {seed}. Generated at git {sha[:12]}{' (dirty)' if dirty else ''}, "
        f"{datetime.now(timezone.utc).date().isoformat()}. `manifest_sha256` is the SHA-256 of "
        "the canonical JSON (sorted keys, no spaces) without that key; SHA256SUMS.txt lists "
        "them. Plan §1.3 defines the splits.\n")
    for name, d in docs.items():
        extra = ""
        if name in ("mvsec_tune", "mvsec_train"):
            extra = f" ranges_s={d['ranges_s_from_first_aps']} aps={d['aps']}"
        print(f"{name}: {d['manifest_sha256'][:12]}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
