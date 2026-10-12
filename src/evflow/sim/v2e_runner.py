"""v2e runner: the only code that invokes v2e (IMPLEMENTATION_PLAN §1.7, `v2e-convert` skill).

v2e runs as a CLI subprocess in its isolated venv (``v2e.python`` / ``v2e.repo`` from
``configs/paths.local.yaml``, D-005); nothing from v2e is imported here. One *unit* = one input
(MVSEC APS frame range, SMD video segment or LaRS clip) converted with one profile and seed:

    input frames ─▶ v2e.py (argv from configs/v2e_profiles.yaml) ─▶ v2e --dvs_h5
                 ─▶ io/v2e_reader (clock map for MVSEC) ─▶ canonical store + manifest

Outputs of a unit in ``<out_dir>/``: ``<unit>.h5`` (canonical store), ``<unit>.v2e.json`` (run
manifest; its SHA-256 is the store's ``simulator_manifest_sha256``), ``<unit>.json`` (final
manifest = run manifest + store facts), ``<unit>.log`` (v2e stdout/stderr). A unit whose store and
final manifest exist is skipped (resumable jobs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
PROFILES = REPO / "configs" / "v2e_profiles.yaml"
PATHS = REPO / "configs" / "paths.local.yaml"

# Flags the runner sets itself; a profile or override may not set them.
RUN_FLAGS = ("--input", "--output_folder", "--dvs_h5", "--dvs_emulator_seed", "--input_frame_rate",
             "--start_time", "--stop_time")
# Flags every run must carry (§1.7).
REQUIRED_COMMON = ("--no_preview", "--skip_video_output")

# v2e log lines that mean the photoreceptor lowpass / timestamp resolution is not honoured
# (v2ecore/v2e_utils.py check_lowpass, v2e.py stage 1). Any of these fails a unit.
UNDERSAMPLING_PATTERNS = (
    r"large IIR mixing factor",
    r"Upsampled src frame intervals of .* is larger than",
    r"will not upsample",
)
NONMONOTONIC_PATTERN = r"nonmonotonic timestamp"
# v2e emulator: a pixel needed more than its per-frame iteration limit (very low thresholds)
TOO_MANY_PATTERN = r"Too many events generated for this frame"
MAX_CLAMPED_FRACTION = 1e-3   # events past the last clock knot that the map would clamp


class PolicyError(ValueError):
    """An argv that violates the v2e argument policy."""


class V2EFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# profiles and argv

def load_profiles(path: str | PathLike = PROFILES) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _norm_args(args) -> dict[str, str | None]:
    """Profile args as an ordered {flag: value-or-None} dict (None = flag without value)."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return {str(k): (None if v is None else str(v)) for k, v in args.items()}
    raise PolicyError(f"profile args must be a mapping flag -> value, got {type(args).__name__}")


def resolve_profile(doc: dict, name: str, overrides: dict | None = None) -> dict:
    """Merged profile ``{"size": [...], "args": {flag: value}, "allow_dvs_params": bool}``.

    ``base:`` is followed recursively; a child's ``size`` replaces the base's, its ``args`` update
    the base's. ``overrides`` (calibration grid cells) may only change flags the profile already
    sets, so a typo cannot silently add a flag."""
    profiles = doc["profiles"]
    if name not in profiles:
        raise KeyError(f"unknown v2e profile {name!r}; known: {sorted(profiles)}")
    p = profiles[name]
    if "base" in p:
        merged = resolve_profile(doc, p["base"])
    else:
        merged = {"size": [], "args": {}, "allow_dvs_params": False}
    if "size" in p:
        merged["size"] = [str(s) for s in p["size"]]
    merged["args"] = {**merged["args"], **_norm_args(p.get("args"))}
    merged["allow_dvs_params"] = bool(p.get("allow_dvs_params", False))
    bad = sorted(set(merged["args"]) & set(RUN_FLAGS))
    if bad:
        raise PolicyError(f"profile {name!r} sets runner-owned flags {bad}")
    for k, v in (overrides or {}).items():
        if k not in merged["args"]:
            raise PolicyError(f"override {k} is not a flag of profile {name!r}")
        merged["args"][k] = str(v)
    return merged


def build_argv(
    doc: dict, profile: str, *, input_path: str, output_folder: str, dvs_h5: str, seed: int,
    input_frame_rate: float, overrides: dict | None = None, start_time: float | None = None,
    stop_time: float | None = None,
) -> list[str]:
    """v2e argv = common + profile size + profile args (+ overrides) + run flags; policy-checked."""
    prof = resolve_profile(doc, profile, overrides)
    common = [str(c) for c in doc["common"]]
    argv = list(common) + list(prof["size"])
    for k, v in prof["args"].items():
        argv += [k] if v is None else [k, v]
    # absolute paths: v2e runs with cwd = its repo
    argv += ["--input", str(Path(input_path).resolve()),
             "--output_folder", str(Path(output_folder).resolve()),
             "--dvs_h5", str(dvs_h5), "--dvs_emulator_seed", str(int(seed)),
             "--input_frame_rate", repr(float(input_frame_rate))]
    if start_time is not None:
        argv += ["--start_time", repr(float(start_time))]
    if stop_time is not None:
        argv += ["--stop_time", repr(float(stop_time))]
    check_policy(argv, allow_dvs_params=prof["allow_dvs_params"])
    return argv


def check_policy(argv: Sequence[str], *, allow_dvs_params: bool = False) -> None:
    flags = [a for a in argv if a.startswith("--")]
    dup = sorted({f for f in flags if flags.count(f) > 1})
    if dup:
        raise PolicyError(f"duplicate v2e flags {dup}")
    for f in REQUIRED_COMMON + ("--dvs_h5", "--dvs_emulator_seed", "--input_frame_rate"):
        if f not in flags:
            raise PolicyError(f"v2e argv lacks required {f}")
    if "--dvs_params" in flags and not allow_dvs_params:
        raise PolicyError("--dvs_params overrides explicit DVS parameters; trial profile only")
    if "--input_frame_rate" in flags:
        fr = float(argv[argv.index("--input_frame_rate") + 1])
        if not np.isfinite(fr) or fr <= 0:
            raise PolicyError(f"bad --input_frame_rate {fr}")


def scan_log(text: str) -> dict:
    """Undersampling warnings (fail a unit) and non-monotonic-timestamp warnings (recorded)."""
    lines = text.splitlines()
    under = [ln.strip() for ln in lines if any(re.search(p, ln) for p in UNDERSAMPLING_PATTERNS)]
    nonmono = [ln.strip() for ln in lines if re.search(NONMONOTONIC_PATTERN, ln)]
    too_many = sum(len(re.findall(TOO_MANY_PATTERN, ln)) for ln in lines)
    return {"undersampling": under, "nonmonotonic": len(nonmono), "too_many_events": too_many}


# ---------------------------------------------------------------------------------------------
# inputs

def sha256_file(path: str | PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_frames(files: Sequence[Path]) -> str:
    """SHA-256 over (name, bytes) of an ordered frame list: identifies a frame-folder input."""
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).name.encode() + b"\0")
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def mvsec_cache_dir(cache_root: str | PathLike, sequence: str, i0: int, i1: int) -> Path:
    """Frame-cache folder of APS frames ``i0..i1`` of ``sequence`` (left camera)."""
    return Path(cache_root) / f"{sequence}_left_aps_{i0:06d}_{i1:06d}"


def export_mvsec_aps(src: str | PathLike, out_dir: str | PathLike, i0: int, i1: int,
                     cam: str = "left") -> dict:
    """Write APS frames ``i0..i1`` (inclusive) of an MVSEC ``*_data.hdf5`` as PNGs to
    ``out_dir/frames/`` and their absolute µs timestamps to ``out_dir/aps_timestamps_us.txt``
    (outside the frame folder: v2e reads every file in it). Idempotent."""
    import cv2
    import h5py

    out = Path(out_dir)
    fdir = out / "frames"
    ts_file = out / "aps_timestamps_us.txt"
    n = i1 - i0 + 1
    if not (ts_file.exists() and fdir.is_dir() and len(list(fdir.glob("*.png"))) == n):
        fdir.mkdir(parents=True, exist_ok=True)
        with h5py.File(src, "r") as f:
            img = f[f"davis/{cam}/image_raw"]
            ts_s = f[f"davis/{cam}/image_raw_ts"][i0 : i1 + 1]
            for k in range(n):
                cv2.imwrite(str(fdir / f"{k:06d}.png"), np.asarray(img[i0 + k]))
        ts_us = np.round(np.asarray(ts_s, np.float64) * 1e6).astype(np.int64)
        np.savetxt(ts_file, ts_us, fmt="%d")
    ts_us = np.loadtxt(ts_file, dtype=np.int64).reshape(-1)
    d = np.diff(ts_us)
    return {"frames_dir": str(fdir), "timestamps_file": str(ts_file), "n_frames": n,
            "aps_first": i0, "aps_last": i1, "timestamps_us": ts_us,
            "fps_measured": 1e6 / float(np.median(d)),
            "frame_interval_us": {"median": float(np.median(d)), "min": int(d.min()),
                                  "max": int(d.max())}}


def export_video_frames(video: str | PathLike, out_dir: str | PathLike, first: int,
                        last: int) -> dict:
    """Decode frames ``first..last`` (inclusive) of a video with ``cv2.VideoCapture`` and write
    them losslessly as PNG to ``out_dir/frames/`` (colour as decoded). v2e is then fed the folder,
    because its video path mis-detects 1080p colour AVIs as monochrome on this OpenCV build and
    skips its RGB→luma step (D-025); the folder path converts exactly as for LaRS. Idempotent."""
    import cv2

    fdir = Path(out_dir) / "frames"
    n = last - first + 1
    if not (fdir.is_dir() and len(list(fdir.glob("*.png"))) == n):
        fdir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        for _ in range(first):
            if not cap.grab():
                raise ValueError(f"{video}: ended before frame {first}")
        for k in range(n):
            ok, im = cap.read()
            if not ok:
                raise ValueError(f"{video}: ended at frame {first + k}")
            cv2.imwrite(str(fdir / f"{k:06d}.png"), im)
        cap.release()
    return {"frames_dir": str(fdir), "first_frame": first, "last_frame": last, "n_frames": n}


def link_frames(files: Sequence[Path], folder: str | PathLike) -> Path:
    """Fresh folder of symlinks ``000000.<ext>``… to ``files`` (v2e reads a whole folder)."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for k, f in enumerate(files):
        os.symlink(Path(f).resolve(), folder / f"{k:06d}{Path(f).suffix}")
    return folder


def ffprobe_rate(video: str | PathLike) -> tuple[float, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    num, den = out.split("/")
    return float(num) / float(den), f"ffprobe r_frame_rate={out}"


# ---------------------------------------------------------------------------------------------
# clock

def uniform_clock_us(n_frames: int, fps: float) -> np.ndarray:
    """v2e's clock for source frame k of a folder/video input: k / fps seconds (v2e.py
    ``start_time = start_frame/srcFps``; interpolated times are normalised to that span)."""
    return np.arange(n_frames, dtype=np.float64) * (1e6 / fps)


def make_clock_map(real_ts_us: np.ndarray, fps: float):
    """Piecewise-linear map from v2e's uniform clock (µs from frame 0) to the real APS clock
    (µs relative to frame 0), knots at the source frames. Returns (map_fn, stats)."""
    real = np.asarray(real_ts_us, np.int64)
    real_rel = (real - real[0]).astype(np.float64)
    uni = uniform_clock_us(len(real), fps)
    dev = real_rel - uni
    stats = {"knots": len(real), "fps": fps,
             "uniform_vs_real_max_abs_dev_us": float(np.max(np.abs(dev))),
             "uniform_vs_real_rms_dev_us": float(np.sqrt(np.mean(dev**2)))}

    def fn(t_us: np.ndarray) -> np.ndarray:
        return np.round(np.interp(np.asarray(t_us, np.float64), uni, real_rel)).astype(np.int64)

    return fn, stats


# ---------------------------------------------------------------------------------------------
# execution

class _GpuMemPoller(threading.Thread):
    """Samples GPU memory.used (MiB) every ``period`` s while v2e runs; peak is a rough figure
    (whole GPU, not per process)."""

    def __init__(self, period: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.period, self.peak, self._halt = period, None, threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.split()
                v = int(out[0])
                self.peak = v if self.peak is None else max(self.peak, v)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._halt.wait(self.period)

    def stop(self) -> int | None:
        self._halt.set()
        self.join(timeout=15)
        return self.peak


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    """True if tracked files changed or untracked files exist under the code/config paths."""
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts",
             "configs", "jobs", "splits"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def redact_home(s: str) -> str:
    home = str(Path.home())
    return s.replace(home, "~")


def run_v2e(argv: list[str], *, v2e_python: str, v2e_repo: str, log_path: Path) -> dict:
    cmd = [v2e_python, str(Path(v2e_repo) / "v2e.py"), *argv]
    poll = _GpuMemPoller()
    poll.start()
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=v2e_repo, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL)
    wall = time.time() - t0
    peak = poll.stop()
    return {"returncode": proc.returncode, "wall_time_s": round(wall, 2),
            "gpu_mem_used_peak_mib": peak}


def run_unit(
    *, unit: str, out_dir: str | PathLike, profile: str, seed: int, input_path: str | PathLike,
    input_sha256: str, input_desc: dict, input_frame_rate: float, frame_rate_source: str,
    width: int, height: int, source: str, sensor: str, t_offset_us: int = 0,
    clock_real_ts_us: np.ndarray | None = None, overrides: dict | None = None,
    start_time: float | None = None, stop_time: float | None = None, keep_raw: bool = False,
    allow_undersampling: bool = False, profiles_doc: dict | None = None,
    paths: dict | None = None, extra: dict | None = None,
) -> dict:
    """Convert one input with one profile; returns the final manifest (skips if done)."""
    from evflow.io.v2e_reader import convert_v2e_h5

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = out / f"{unit}.h5"
    final = out / f"{unit}.json"
    if store.exists() and final.exists():
        return {**json.loads(final.read_text()), "skipped": True}

    doc = profiles_doc or load_profiles()
    paths = paths or yaml.safe_load(PATHS.read_text())
    work = out / f"{unit}.v2e"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    argv = build_argv(doc, profile, input_path=str(input_path), output_folder=str(work),
                      dvs_h5=f"{unit}.h5", seed=seed, input_frame_rate=input_frame_rate,
                      overrides=overrides, start_time=start_time, stop_time=stop_time)
    log_path = out / f"{unit}.log"
    res = run_v2e(argv, v2e_python=paths["v2e"]["python"], v2e_repo=paths["v2e"]["repo"],
                  log_path=log_path)
    warn = scan_log(log_path.read_text(errors="replace"))
    raw = work / f"{unit}.h5"
    if res["returncode"] != 0 or not raw.exists():
        raise V2EFailed(f"{unit}: v2e exit {res['returncode']}, see {log_path}")
    if warn["undersampling"] and not allow_undersampling:
        raise V2EFailed(f"{unit}: lowpass/timestamp undersampling warning: "
                        f"{warn['undersampling'][0]}")

    commit = (REPO / doc["v2e_commit_file"]).read_text().strip()
    run_manifest = {
        "kind": "v2e_run_manifest",
        "unit": unit, "profile": profile, "overrides": overrides or {}, "seed": int(seed),
        "v2e_commit": commit, "v2e_argv": [redact_home(a) for a in argv],
        "input": {**input_desc, "sha256": input_sha256},
        "input_frame_rate": input_frame_rate, "frame_rate_source": frame_rate_source,
        "v2e_h5_sha256": sha256_file(raw), "v2e_h5_bytes": raw.stat().st_size,
        "wall_time_s": res["wall_time_s"], "gpu": gpu_name(),
        "gpu_mem_used_peak_mib": res["gpu_mem_used_peak_mib"],
        "warnings": warn, "allow_undersampling": bool(allow_undersampling),
        "git_sha": git_sha(), "git_dirty": git_dirty(),
        "created_by": "src/evflow/sim/v2e_runner.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    time_map, clock, domain_end = None, None, None
    if clock_real_ts_us is not None:
        time_map, clock = make_clock_map(clock_real_ts_us, input_frame_rate)
        domain_end = float(uniform_clock_us(len(clock_real_ts_us), input_frame_rate)[-1])
        run_manifest["clock_map"] = {"kind": "piecewise_linear_v2e_uniform_to_aps", **clock}
    rm_path = out / f"{unit}.v2e.json"
    rm_path.write_text(json.dumps(run_manifest, indent=1) + "\n")
    rm_sha = sha256_file(rm_path)

    conv = convert_v2e_h5(raw, store, width=width, height=height, source=source, sensor=sensor,
                          t_offset_us=t_offset_us, time_map=time_map,
                          time_map_domain_end_us=domain_end, simulator_manifest_sha256=rm_sha)
    clamped = conv["n_events_beyond_clock_map"]
    if clamped and clamped > MAX_CLAMPED_FRACTION * max(conv["n_events"], 1):
        raise V2EFailed(f"{unit}: {clamped} events after the last APS frame would be clamped")
    final_doc = {**run_manifest, "kind": "v2e_unit_manifest", "run_manifest_sha256": rm_sha,
                 "store": redact_home(str(store)), "store_sha256": sha256_file(store),
                 "store_bytes": store.stat().st_size, "conversion": conv}
    if not keep_raw:
        shutil.rmtree(work)
        final_doc["v2e_h5_deleted_after_conversion"] = True
    final.write_text(json.dumps(final_doc, indent=1) + "\n")
    return final_doc


# ---------------------------------------------------------------------------------------------
# CLI (used by jobs/*.sh; one unit per call)

def _parse_overrides(items: Sequence[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not k.startswith("--") or not v:
            raise SystemExit(f"bad --set {it!r}; use --set --flag=value")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one v2e unit (see module docstring).")
    sub = ap.add_subparsers(dest="source", required=True)
    for name in ("mvsec", "smd", "lars"):
        s = sub.add_parser(name)
        s.add_argument("--unit", required=True)
        s.add_argument("--out", required=True, type=Path)
        s.add_argument("--profile", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--set", action="append", default=[], metavar="--FLAG=VALUE",
                       help="override a flag the profile sets (calibration grid)")
        s.add_argument("--keep-raw", action="store_true")
        s.add_argument("--allow-undersampling", action="store_true",
                       help="record instead of fail (throughput sweep only)")
        if name == "mvsec":
            s.add_argument("--sequence", required=True)
            s.add_argument("--aps", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
            s.add_argument("--split", required=True,
                           help="split manifest the APS range must lie in (sim/splits.py)")
            s.add_argument("--lead-frames", type=int, default=0,
                           help="frames before the split range (evaluation splits only, D-032)")
            s.add_argument("--frames-cache", type=Path, required=True)
        elif name == "smd":
            s.add_argument("--video", required=True)
            s.add_argument("--start", type=float, required=True)
            s.add_argument("--stop", type=float, required=True)
            s.add_argument("--frames-cache", type=Path, required=True)
        else:
            s.add_argument("--subset", required=True)
            s.add_argument("--clip", required=True, help="prefix:first:last")
    a = ap.parse_args(argv)
    paths = yaml.safe_load(PATHS.read_text())
    doc = load_profiles()
    over = _parse_overrides(a.set)
    common = dict(unit=a.unit, out_dir=a.out, profile=a.profile, seed=a.seed, overrides=over,
                  keep_raw=a.keep_raw, allow_undersampling=a.allow_undersampling,
                  profiles_doc=doc, paths=paths)
    size = resolve_profile(doc, a.profile)["size"]

    if a.source == "mvsec":
        from evflow.io.mvsec import HEIGHT, SENSOR, WIDTH
        from evflow.sim.splits import check_aps_in_split, load_split

        split = load_split(a.split)
        check_aps_in_split(split, a.sequence, a.aps[0], a.aps[1], a.lead_frames)
        scene = re.sub(r"\d+$", "", a.sequence)
        src = Path(paths["mvsec"]["root"]) / scene / f"{a.sequence}_data.hdf5"
        cache = mvsec_cache_dir(a.frames_cache, a.sequence, a.aps[0], a.aps[1])
        ex = export_mvsec_aps(src, cache, a.aps[0], a.aps[1])
        files = sorted(Path(ex["frames_dir"]).glob("*.png"))
        ts = ex.pop("timestamps_us")
        desc = {"dataset": "mvsec", "sequence": a.sequence, "camera": "left",
                "split": a.split, "split_manifest_sha256": split["manifest_sha256"],
                "lead_frames": a.lead_frames,
                "source_file": redact_home(str(src)), **{k: v for k, v in ex.items()
                                                         if k not in ("frames_dir",)}}
        desc["frames_dir"] = redact_home(ex["frames_dir"])
        desc["timestamps_file"] = redact_home(ex["timestamps_file"])
        doc_ = run_unit(**common, input_path=ex["frames_dir"], input_sha256=sha256_frames(files),
                        input_desc=desc, input_frame_rate=ex["fps_measured"],
                        frame_rate_source="1e6 / median(diff(APS image_raw_ts)) of the exported "
                                          "range; events re-mapped to the real APS clock",
                        width=WIDTH, height=HEIGHT, source=f"v2e/mvsec/{a.sequence}/left_aps",
                        sensor=f"v2e({SENSOR} APS)", t_offset_us=int(ts[0]),
                        clock_real_ts_us=ts)
    elif a.source == "smd":
        video = Path(paths["smd"]["onboard"]) / "Videos" / a.video
        fps, why = ffprobe_rate(video)
        w, h = _size_from_args(size)
        f0, f1 = int(round(a.start * fps)), int(round(a.stop * fps))
        cache = a.frames_cache / f"smd_{Path(a.video).stem}_{f0:06d}_{f1:06d}"
        ex = export_video_frames(video, cache, f0, f1)
        files = sorted(Path(ex["frames_dir"]).glob("*.png"))
        from evflow.io.frames import native_size, scale_factors

        nat = native_size(files[0])
        desc = {"dataset": "smd", "video": a.video, "video_sha256": sha256_file(video),
                "segment_s": [a.start, a.stop], "n_frames": ex["n_frames"],
                "first_frame": f0, "last_frame": f1,
                "frames_dir": redact_home(str(Path(ex["frames_dir"]).resolve())),
                "input_route": "frames decoded by cv2.VideoCapture -> lossless PNG folder (D-025)",
                "native_size": list(nat), "output_size": [w, h],
                "scale_xy": list(scale_factors(nat, (w, h)))}
        doc_ = run_unit(**common, input_path=ex["frames_dir"], input_sha256=sha256_frames(files),
                        input_desc=desc, input_frame_rate=fps, frame_rate_source=why,
                        width=w, height=h, source=f"v2e/smd/{Path(a.video).stem}",
                        sensor="v2e(SMD VIS_Onboard video)", t_offset_us=int(round(
                            f0 * 1e6 / fps)))
    else:
        from evflow.io.frames import native_size, scale_factors
        from evflow.io.lars import clip_frames

        prefix, first, last = a.clip.rsplit(":", 2)
        files = clip_frames(paths["lars"]["root"], a.subset, prefix, int(first), int(last))
        folder = link_frames(files, Path(a.out) / f"{a.unit}.frames")
        fps = float(doc["lars"]["assumed_fps"])
        w, h = _size_from_args(size)
        nat = native_size(files[0])
        desc = {"dataset": "lars", "subset": a.subset, "clip": a.clip, "n_frames": len(files),
                "native_size": list(nat), "output_size": [w, h],
                "scale_xy": list(scale_factors(nat, (w, h))),
                "resize": "v2e cv2.resize INTER_AREA to output size, aspect not kept (D-054)"}
        doc_ = run_unit(**common, input_path=folder, input_sha256=sha256_frames(files),
                        input_desc=desc, input_frame_rate=fps,
                        frame_rate_source=f"assumed ({doc['lars']['assumed_fps_decision']})",
                        width=w, height=h, source=f"v2e/lars/{a.subset}/{prefix}",
                        sensor="v2e(LaRS images_seq)")
        shutil.rmtree(folder, ignore_errors=True)
    summary = {k: doc_.get(k) for k in ("unit", "wall_time_s", "skipped")}
    summary["n_events"] = doc_["conversion"]["n_events"]
    print(json.dumps(summary))
    return 0


def _size_from_args(size: list[str]) -> tuple[int, int]:
    if "--dvs346" in size:
        return 346, 260
    if "--output_width" in size:
        return int(size[size.index("--output_width") + 1]), int(
            size[size.index("--output_height") + 1])
    raise PolicyError(f"profile size {size} does not fix the output size")


if __name__ == "__main__":
    sys.exit(main())
