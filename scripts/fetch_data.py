"""Download, extract and sanity-check the datasets listed in configs/data_sources.yaml.

Phase 0, plan step 6 (data access); decisions D-011..D-013. Everything lands under the data root
(default <repo>/data, gitignored). Resumable and idempotent: finished downloads (right size) and
finished extractions (marker file) are skipped. The committed record of what is on disk is
results/data_inventory.json from scripts/check_data.py, not this script's log.

    python -I scripts/fetch_data.py [--only mvsec,dsec] [--kind http] [--no-extract]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import yaml

REPO = Path(__file__).resolve().parents[1]
GDRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
REQUIRED = ("dataset", "name", "kind", "dest")
KINDS = ("http", "gdrive", "local")
EXTRACTS = ("zip", "tar")


class FetchError(RuntimeError):
    pass


# ---------------------------------------------------------------- config


def load_sources(path: Path) -> list[dict]:
    sources = yaml.safe_load(path.read_text())["sources"]
    validate_sources(sources)
    return sources


def validate_sources(sources: list[dict]) -> None:
    dests = set()
    for s in sources:
        missing = [k for k in REQUIRED if k not in s]
        if missing:
            raise ValueError(f"{s.get('name')}: missing {missing}")
        if s["kind"] not in KINDS:
            raise ValueError(f"{s['name']}: kind {s['kind']!r} not in {KINDS}")
        if s["kind"] == "http" and "url" not in s:
            raise ValueError(f"{s['name']}: http source needs url")
        if s["kind"] == "gdrive" and "gdrive_id" not in s:
            raise ValueError(f"{s['name']}: gdrive source needs gdrive_id")
        if s.get("extract", "none") not in (*EXTRACTS, "none"):
            raise ValueError(f"{s['name']}: extract {s['extract']!r}")
        if s.get("extract", "none") != "none" and "extract_to" not in s:
            raise ValueError(f"{s['name']}: extracted source needs extract_to")
        for key in ("dest", "extract_to"):
            if key in s and not is_safe_relpath(s[key]):
                raise ValueError(f"{s['name']}: unsafe {key} {s[key]!r}")
        if s["dest"] in dests:
            raise ValueError(f"duplicate dest {s['dest']}")
        dests.add(s["dest"])


def is_safe_relpath(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return bool(name) and not p.is_absolute() and ".." not in p.parts and ":" not in p.parts[0]


# ---------------------------------------------------------------- download


def is_complete(dest: Path, expected_bytes: int | None) -> bool:
    if not dest.exists():
        return False
    return expected_bytes is None or dest.stat().st_size == expected_bytes


def curl(url: str, part: Path) -> None:
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-", "-sS",
           "-o", str(part), url]
    subprocess.run(cmd, check=True)


def download_http(src: dict, part: Path) -> None:
    curl(src["url"], part)


def gdrive_url(src: dict) -> str:
    url = GDRIVE_URL.format(id=src["gdrive_id"])
    return url + (f"&resourcekey={src['resourcekey']}" if "resourcekey" in src else "")


def download_gdrive(src: dict, part: Path) -> None:
    """Direct, resumable Drive download (confirm=t skips the virus-scan page).

    gdown was dropped: it cannot pass resourcekeys of old 0B... ids, and both tools hit the same
    per-file quota. A quota or permission refusal comes back as an HTML page, which is rejected.
    """
    curl(gdrive_url(src), part)
    with open(part, "rb") as f:
        head = f.read(512).lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        part.unlink()
        raise FetchError(f"{src['name']}: Drive returned an HTML page (quota or permission)")


def marker_path(src: dict, root: Path) -> Path:
    return root / src["extract_to"] / f".extracted_from_{src['name']}"


def fetch(src: dict, root: Path, downloader=None) -> str:
    dest = root / src["dest"]
    expected = src.get("expected_bytes")
    if is_complete(dest, expected):
        return "present"
    if src.get("delete_archive") and marker_path(src, root).exists():
        return "extracted-earlier"
    if src["kind"] == "local":
        raise FetchError(f"{src['name']}: local file missing or wrong size at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if downloader is None:
        downloader = download_http if src["kind"] == "http" else download_gdrive
    downloader(src, part)
    if expected is not None and part.stat().st_size != expected:
        raise FetchError(f"{src['name']}: got {part.stat().st_size} B, expected {expected} B")
    part.rename(dest)
    return "downloaded"


# ---------------------------------------------------------------- extract


def unsafe_members(archive: Path) -> list[str]:
    """Members that would escape the target directory or are links/devices."""
    bad = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                is_link = (info.external_attr >> 16) & 0o170000 == 0o120000
                if is_link or not is_safe_relpath(info.filename):
                    bad.append(info.filename)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not (m.isfile() or m.isdir()) or not is_safe_relpath(m.name):
                    bad.append(m.name)
    return bad


def extract(src: dict, root: Path) -> str:
    archive, target = root / src["dest"], root / src["extract_to"]
    marker = marker_path(src, root)
    if marker.exists():
        return "extracted-earlier"
    bad = unsafe_members(archive)
    if bad:
        raise FetchError(f"{src['name']}: unsafe archive members {bad[:5]}")
    target.mkdir(parents=True, exist_ok=True)
    if src["extract"] == "zip":
        with zipfile.ZipFile(archive) as z:  # CRC is checked as each member is read
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")
    marker.write_text(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds") + "\n")
    if src.get("delete_archive"):
        archive.unlink()
    return "extracted"


# ---------------------------------------------------------------- verify


def verify(src: dict, root: Path) -> str:
    dest = root / src["dest"]
    if not dest.exists():  # archive deleted after extraction
        return "archive-removed"
    if dest.suffix in (".hdf5", ".h5"):
        import h5py

        with h5py.File(dest, "r") as f:
            return "h5 keys: " + ",".join(sorted(f.keys()))
    if dest.suffix == ".npz":
        with zipfile.ZipFile(dest) as z:
            bad = z.testzip()
            if bad:
                raise FetchError(f"{src['name']}: corrupt member {bad}")
            return "npz arrays: " + ",".join(sorted(n.removesuffix(".npy") for n in z.namelist()))
    return f"{dest.stat().st_size} B"


# ---------------------------------------------------------------- main


def run(sources: list[dict], root: Path, do_extract: bool, log_path: Path) -> list[dict]:
    records = []
    for src in sources:
        rec = {"name": src["name"], "dataset": src["dataset"], "dest": src["dest"]}
        t0 = time.time()
        try:
            rec["download"] = fetch(src, root)
            if do_extract and src.get("extract", "none") != "none":
                rec["extract"] = extract(src, root)
            rec["verify"] = verify(src, root)
            rec["ok"] = True
        except (FetchError, subprocess.CalledProcessError, OSError, zipfile.BadZipFile) as e:
            rec["ok"], rec["error"] = False, str(e)[:500]
        rec["seconds"] = round(time.time() - t0, 1)
        rec["at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=REPO / "configs/data_sources.yaml")
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument("--only", default="", help="comma-separated datasets (default: all)")
    ap.add_argument("--kind", choices=KINDS, help="only sources of this kind")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args(argv)

    sources = load_sources(args.sources)
    if args.only:
        wanted = set(args.only.split(","))
        sources = [s for s in sources if s["dataset"] in wanted]
    if args.kind:
        sources = [s for s in sources if s["kind"] == args.kind]
    args.data_root.mkdir(parents=True, exist_ok=True)
    records = run(sources, args.data_root, not args.no_extract, args.data_root / "_fetch_log.jsonl")
    failed = [r["name"] for r in records if not r["ok"]]
    print(f"{len(records) - len(failed)}/{len(records)} ok", f"failed: {failed}" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
