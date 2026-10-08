"""Convert one MVSEC or DSEC sequence into the canonical event store (Phase 1, plan step 1).

    python scripts/convert_events.py mvsec --sequence indoor_flying1 [--cam left]
    python scripts/convert_events.py dsec  --sequence thun_00_a

Inputs come from configs/paths.local.yaml; the store is written to ``outputs.events``
(``data/converted/``, gitignored, D-018) as ``<dataset>_<sequence>_<cam>.h5`` with a JSON manifest
next to it (source files with size and SHA-256, converter summary, git state).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from check_data import git_state, redact_home, sha256_file

from evflow.io import dsec, mvsec

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=["mvsec", "dsec"])
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--cam", default="left")
    ap.add_argument("--paths", type=Path, default=REPO / "configs/paths.local.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.paths.read_text())
    out_dir = Path(cfg["outputs"]["events"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{args.dataset}_{args.sequence}_{args.cam}.h5"
    if dst.exists() and not args.overwrite:
        print(f"{dst} exists (use --overwrite)", file=sys.stderr)
        return 1

    t0 = time.time()
    if args.dataset == "mvsec":
        scene = re.sub(r"\d+$", "", args.sequence)
        src = Path(cfg["mvsec"]["root"]) / scene / f"{args.sequence}_data.hdf5"
        sources = [src]
        summary = mvsec.convert_events(src, dst, sequence=args.sequence, cam=args.cam)
    else:
        seq_dir = Path(cfg["dsec"]["root"]) / args.sequence
        ev_dir = seq_dir / f"events_{args.cam}"
        sources = [ev_dir / "events.h5", ev_dir / "rectify_map.h5"]
        summary = dsec.convert_events(seq_dir, dst, sequence=args.sequence, camera=args.cam)
    wall = time.time() - t0

    sha, dirty = git_state()
    manifest = {
        "kind": "event_store_manifest",
        "dataset": args.dataset,
        "sequence": args.sequence,
        "camera": args.cam,
        "store": redact_home(dst),
        "store_size_bytes": dst.stat().st_size,
        "sources": [{"path": redact_home(p), "size_bytes": p.stat().st_size,
                     "sha256": sha256_file(p)} for p in sources],
        "summary": summary,
        "wall_time_s": round(wall, 1),
        "git_sha": sha,
        "git_dirty": dirty,
        "created_by": "scripts/convert_events.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    dst.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"store": manifest["store"], **summary, "wall_time_s": round(wall, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
