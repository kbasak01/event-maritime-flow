"""Per-window GT cache for Phase 5 training and selection (``mvsec_train`` clips, ``mvsec_tune``).

GT per window = ``io.mvsec.gt_flow_for_window`` (EV-FlowNet propagation, D-019) over
[aps[k], aps[k + dt]] for every window ``evflow.train.data`` uses (D-026 clips / the tune slice,
first ``warmup`` windows dropped). Only ``mvsec_train`` / ``mvsec_tune`` are accepted (split
guard). Writes ``<outputs.events>/gt_cache/<split>_dt<k>.h5`` (``aps_start``, ``t_start_us``,
``t_end_us``, ``flow`` float32 (N, 2, H, W), ``valid`` bool (N, H, W)) and a JSON manifest next to
it (source npz SHA from the split manifest, git sha, SHA-256 of the h5). Skips a cache whose
manifest says it was built from a clean tree; rebuilds one built from a dirty tree.

    python scripts/cache_mvsec_gt.py --split mvsec_train --dt 1 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml
from check_data import sha256_file

from evflow.io import mvsec as mvsec_io
from evflow.sim.splits import load_split
from evflow.sim.v2e_runner import git_dirty, git_sha
from evflow.train.data import (
    ALLOWED_SPLITS,
    HEIGHT,
    SEQUENCE,
    WIDTH,
    clip_window_starts,
    gt_cache_path,
    guard_split,
    train_clips_from_split,
    tune_clip,
)

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    ap.add_argument("--dt", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: configs/evflow.yaml train.warmup_windows")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    paths = yaml.safe_load((REPO / "configs" / "paths.local.yaml").read_text())
    evcfg = yaml.safe_load((REPO / "configs" / "evflow.yaml").read_text())
    corpus_cfg = yaml.safe_load((REPO / "configs" / "v2e_corpus_mvsec.yaml").read_text())
    warmup = a.warmup if a.warmup is not None else int(evcfg["train"]["warmup_windows"])
    out_root = Path(paths["outputs"]["events"]) / "gt_cache"
    out_root.mkdir(parents=True, exist_ok=True)
    src = Path(paths["mvsec"]["root"]) / "outdoor_day" / f"{SEQUENCE}_data.hdf5"
    npz = Path(paths["mvsec"]["gt_flow"]) / "outdoor_day" / f"{SEQUENCE}_gt_flow_dist.npz"
    aps = mvsec_io.read_aps_timestamps_us(src)
    split = load_split(a.split)
    clips = (train_clips_from_split(split, aps, corpus_cfg) if a.split == "mvsec_train"
             else [tune_clip(split)])
    for c in clips:
        guard_split(a.split, c)
    npz_sha = next((s["sha256"] for s in split["sources"] if s["path"].endswith(npz.name)), None)
    gt = mvsec_io.GTFlow(npz)

    for dt in a.dt:
        out = gt_cache_path(out_root, a.split, dt)
        man = out.with_suffix(".json")
        if out.exists() and man.exists() and not a.force:
            if not json.loads(man.read_text()).get("git_dirty", True):
                print(f"{out.name}: present")
                continue
            print(f"{out.name}: present but built from a dirty tree, rebuilding")
        ks = np.concatenate([clip_window_starts(c, dt, warmup) for c in clips])
        tmp = out.with_suffix(".h5.tmp")
        t_wall = time.time()
        n_valid = np.zeros(ks.size, np.int64)
        with h5py.File(tmp, "w") as f:
            f.attrs.update(split=a.split, dt=dt, sequence=SEQUENCE, warmup=warmup)
            f["aps_start"] = ks
            f["t_start_us"] = aps[ks]
            f["t_end_us"] = aps[ks + dt]
            fl = f.create_dataset("flow", (ks.size, 2, HEIGHT, WIDTH), np.float32,
                                  chunks=(1, 2, HEIGHT, WIDTH))
            va = f.create_dataset("valid", (ks.size, HEIGHT, WIDTH), bool,
                                  chunks=(1, HEIGHT, WIDTH))
            for i, k in enumerate(ks):
                flow, valid = mvsec_io.gt_flow_for_window(gt, int(aps[k]), int(aps[k + dt]))
                fl[i] = flow.astype(np.float32)
                va[i] = valid
                n_valid[i] = int(valid.sum())
                if i % 2000 == 0:
                    print(f"{out.name}: {i}/{ks.size}", flush=True)
        tmp.replace(out)
        doc = {"kind": "gt_cache", "split": a.split, "split_manifest_sha256":
               split["manifest_sha256"], "sequence": SEQUENCE, "dt": dt, "warmup": warmup,
               "clips": [[c.aps_first, c.aps_last] for c in clips], "n_windows": int(ks.size),
               "aps_start_first_last": [int(ks[0]), int(ks[-1])],
               "mean_valid_px": float(n_valid.mean()), "source_npz": npz.name,
               "source_npz_sha256": npz_sha, "gt_rule": "io.mvsec.gt_flow_for_window (D-019)",
               "sha256": sha256_file(out), "wall_time_s": round(time.time() - t_wall, 1),
               "git_sha": git_sha(), "git_dirty": git_dirty(),
               "created_by": "scripts/cache_mvsec_gt.py",
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        man.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{out.name}: {ks.size} windows, {doc['wall_time_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
