"""Phase 4 reproduction gate, part (a)+(b): E-RAFT's OWN MVSEC 20 Hz evaluation pipeline.

Runs in the isolated venv ``external/venvs/eraft`` (E-RAFT's loader needs pandas + PyTables and
its test module imports scikit-image/OpenCV; D-039) against the pinned clone ``external/E-RAFT``
(c58ce05), which is never modified:

    external/venvs/eraft/bin/python scripts/eraft_reference_run.py \
        --eraft external/E-RAFT --data data/eraft/mvsec_outdoor_day_1_20Hz \
        --checkpoint data/eraft/checkpoints/mvsec_20.tar --out runs/phase4-baselines/repro

(a) E-RAFT's ``config/mvsec_20.json`` + ``MvsecFlowRecurrent`` loader (dense mask, 256x256
centre crop, its own voxel transform) + ``ERAFT`` + its ``TestRaftEventsWarm`` test loop (warm
start, state reset on index jumps), driven exactly as ``main.py`` does; a stub visualizer captures
``batch[-1]`` (flow_est, flow, gt_valid_mask, idx). The sparse mask is E-RAFT's own
``evaluation_type = 'sparse'`` branch, set on the dataset instance in a second pass.
(b) "E-RAFT's own evaluation" (the repo ships no metric code; user decision 2026-10-09): RAFT's
``validate_kitti`` EPE formula -- per sample ``sqrt(sum((est - gt)^2))`` averaged over the pixels
with mask >= 0.5, then the mean over samples -- plus the pooled variant and 1PE/3PE (% of masked
pixels with EPE > 1 / > 3 px). Written here, independently of ``evflow`` (which this env lacks).

Outputs ``reference.npz`` (per-sample arrays) and ``reference.json`` (metrics and provenance) in
``--out`` (gitignored runs/); scripts/eraft_reproduce.py turns them into the results JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raft_eval(est: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """RAFT validate_kitti-style: per-sample masked mean EPE, then mean over samples."""
    import torch

    per_epe, per_1pe, per_3pe, pooled = [], [], [], []
    for e, g, m in zip(est, gt, mask, strict=True):
        epe = torch.sum((torch.from_numpy(e).double() - torch.from_numpy(g).double()) ** 2,
                        dim=0).sqrt().view(-1)
        val = torch.from_numpy(m).view(-1) >= 0.5
        v = epe[val]
        if v.numel() == 0:
            continue
        per_epe.append(v.mean().item())
        per_1pe.append(100.0 * (v > 1).double().mean().item())
        per_3pe.append(100.0 * (v > 3).double().mean().item())
        pooled.append(v.numpy())
    pooled = np.concatenate(pooled)
    return {"epe": float(np.mean(per_epe)), "npe_1": float(np.mean(per_1pe)),
            "npe_3": float(np.mean(per_3pe)), "epe_pooled": float(pooled.mean()),
            "npe_1_pooled": float(100.0 * (pooled > 1).mean()),
            "npe_3_pooled": float(100.0 * (pooled > 3).mean()),
            "n_samples": len(per_epe), "per_sample_epe": per_epe}


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line, verbose=False):
        self.lines.append(str(line))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eraft", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    eraft = args.eraft.resolve()
    data = args.data.resolve()
    ckpt = args.checkpoint.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=eraft, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=eraft,
                           capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{eraft} has local modifications; the reference clone must be pristine")

    sys.path.insert(0, str(eraft))
    os.chdir(out)  # E-RAFT's helpers may create folders relative to the cwd
    import torch
    from loader.loader_mvsec_flow import MvsecFlowRecurrent
    from model import eraft as eraft_model
    from test import TestRaftEventsWarm
    from torch.utils.data import DataLoader

    config = json.loads((eraft / "config/mvsec_20.json").read_text())
    largs = config["data_loader"]["test"]["args"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    captured: list[dict] = []

    class Capture:
        def __init__(self, data_loader, save_path, additional_args=None):
            pass

        def __call__(self, sample, *a, **k):
            captured.append({
                "idx": int(sample["idx"].item()),
                "flow_est": sample["flow_est"][0].detach().cpu().numpy().astype(np.float32),
                "flow": sample["flow"][0].detach().cpu().numpy().astype(np.float32),
                "mask": sample["gt_valid_mask"][0, 0].detach().cpu().numpy() >= 0.5,
            })

    # (a) exactly main.py's MVSEC warm-start path
    test_set = MvsecFlowRecurrent(args=largs, type="test", path=str(data))
    loader = DataLoader(test_set, batch_size=largs["batch_size"], shuffle=largs["shuffle"],
                        num_workers=0, drop_last=True)
    model = eraft_model.ERAFT(config=config, n_first_channels=largs["num_voxel_bins"])
    checkpoint = torch.load(ckpt, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    logger = _Logger()
    tester = TestRaftEventsWarm(model=model, config=config, data_loader=loader,
                                test_logger=logger, save_path=str(out), visualizer=Capture,
                                additional_args=None)
    t0 = time.time()
    tester._test()
    wall = time.time() - t0

    # E-RAFT's sparse branch: same loader, evaluation_type switched on the instance
    ds = test_set.dataset
    ds.evaluation_type = "sparse"
    sparse = []
    for i in range(len(ds)):
        s = ds[i]
        sparse.append(s["gt_valid_mask"][0].cpu().numpy().astype(bool))
    ds.evaluation_type = "dense"

    idx = np.array([c["idx"] for c in captured], np.int64)
    est = np.stack([c["flow_est"] for c in captured])
    gt = np.stack([c["flow"] for c in captured])
    dense = np.stack([c["mask"] for c in captured])
    sparse = np.stack(sparse)
    if len(sparse) != len(idx):
        raise SystemExit(f"{len(sparse)} sparse masks for {len(idx)} samples")
    np.savez_compressed(out / "reference.npz", idx=idx, flow_est=est, flow_gt=gt,
                        valid_dense=dense, valid_sparse=sparse)

    doc = {
        "kind": "eraft_reference_run",
        "eraft_commit": commit,
        "config": "config/mvsec_20.json",
        "config_contents": config,
        "data": str(data),
        "checkpoint_sha256": sha256_file(ckpt),
        "n_samples": int(len(idx)),
        "idx_range": [int(idx.min()), int(idx.max())],
        "resets_logged": sum("Resetting" in ln for ln in logger.lines),
        "evaluation": "RAFT validate_kitti formula (per-sample masked mean EPE, then mean)",
        "dense": {k: v for k, v in raft_eval(est, gt, dense).items()},
        "sparse": {k: v for k, v in raft_eval(est, gt, sparse).items()},
        "wall_time_s": round(wall, 1),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "reference_npz_sha256": sha256_file(out / "reference.npz"),
    }
    (out / "reference.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"E-RAFT reference: {len(idx)} samples, dense EPE {doc['dense']['epe']:.4f}, "
          f"sparse EPE {doc['sparse']['epe']:.4f} ({wall:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
