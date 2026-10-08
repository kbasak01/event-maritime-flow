"""Inventory the datasets named in configs/paths.local.yaml (Phase 0, plan step 6).

For every data path: does it exist, how many files, how many bytes, and SHA-256 of the small files
(GT npz, flow timestamps, annotation zips, checkpoints under the size cap). Large files get size
only. Writes results/data_inventory.json; a missing dataset is recorded as absent, never an error.

    python scripts/check_data.py [--paths configs/paths.local.yaml]
                                 [--out results/data_inventory.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# Keys in paths.local.yaml that are not datasets (tools, environments, output areas).
NON_DATA_KEYS = ("outputs", "v2e.repo", "v2e.python")
SKIP_SUFFIXES = (":Zone.Identifier",)


def flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    """Nested YAML mapping -> {"dotted.key": "path"} for every string leaf."""
    out: dict[str, str] = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif isinstance(value, str):
            out[name] = value
    return out


def is_data_key(name: str) -> bool:
    return not any(name == k or name.startswith(k + ".") for k in NON_DATA_KEYS)


def redact_home(path: Path) -> str:
    home = str(Path.home())
    s = str(path)
    return "~" + s[len(home):] if s == home or s.startswith(home + os.sep) else s


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith(".") or fn.endswith(SKIP_SUFFIXES):
                continue
            files.append(Path(dirpath) / fn)
    return files


def inventory_entry(raw_path: str, sha_max_bytes: int, max_files: int) -> dict:
    root = Path(raw_path).expanduser()
    entry: dict = {"path": redact_home(root), "present": root.exists()}
    if not root.exists():
        return entry
    files = list_files(root)
    base = root.parent if root.is_file() else root
    sizes = [f.stat().st_size for f in files]
    entry.update(
        kind="file" if root.is_file() else "dir",
        n_files=len(files),
        total_bytes=int(sum(sizes)),
        files_listed=min(len(files), max_files),
    )
    listed = []
    for f, size in zip(files[:max_files], sizes[:max_files], strict=True):
        rec = {"rel": str(f.relative_to(base)), "bytes": size}
        rec["sha256"] = sha256_file(f) if size <= sha_max_bytes else None
        listed.append(rec)
    entry["files"] = listed
    if len(files) > max_files:
        counts: dict[str, int] = {}
        for f in files[max_files:]:
            top = f.relative_to(base).parts[0]
            counts[top] = counts.get(top, 0) + 1
        entry["unlisted_counts_by_top_dir"] = dict(sorted(counts.items()))
    return entry


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def build_inventory(paths_cfg: dict, sha_max_bytes: int, max_files: int) -> dict:
    sha, dirty = git_state()
    datasets = {
        name: inventory_entry(p, sha_max_bytes, max_files)
        for name, p in sorted(flatten(paths_cfg).items())
        if is_data_key(name)
    }
    return {
        "kind": "data_inventory",
        "created_by": "scripts/check_data.py",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "sha_max_bytes": sha_max_bytes,
        "max_files_listed": max_files,
        "summary": {
            "present": sorted(k for k, v in datasets.items() if v["present"]),
            "absent": sorted(k for k, v in datasets.items() if not v["present"]),
        },
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=Path, default=REPO / "configs/paths.local.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "results/data_inventory.json")
    ap.add_argument("--sha-max-mb", type=float, default=64.0)
    ap.add_argument("--max-files", type=int, default=2000)
    args = ap.parse_args(argv)

    if not args.paths.exists():
        print(f"missing {args.paths}; copy configs/paths.local.yaml.example", file=sys.stderr)
        return 2
    paths_cfg = yaml.safe_load(args.paths.read_text()) or {}
    inv = build_inventory(paths_cfg, int(args.sha_max_mb * 2**20), args.max_files)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n")
    for name, e in inv["datasets"].items():
        state = "ABSENT"
        if e["present"]:
            state = f"{e['n_files']} files, {e['total_bytes'] / 2**30:.2f} GiB"
        print(f"{name:32s} {state:28s} {e['path']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
