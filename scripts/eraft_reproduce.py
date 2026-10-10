"""Phase 4 reproduction gate, parts (c)-(e): our metric code and our input pipeline against E-RAFT.

    python scripts/eraft_reproduce.py --reference runs/phase4-baselines/repro

Reads the output of scripts/eraft_reference_run.py (E-RAFT's own pipeline + "E-RAFT's own
evaluation", run in external/venvs/eraft) and writes results/baselines/eraft_reproduction.json:

(c) **Metric parity.** ``evflow.flow.metrics.flow_metrics`` on E-RAFT's flows, with the masks built
    by us from configs/eval_mvsec.yaml ``reproduction_eraft`` (GT = optical_flow/{i:06d}.npy as-is;
    valid = (u != 0) | (v != 0) and full-image rows < 193; region center_256; sparse: AND pixel has
    an event in E-RAFT's "new" event file i+1). Pass: |ours - reference| <= match_tolerance_px per
    sample and on the mean, for dense and sparse; our masks must equal E-RAFT's.
(d) **Input pipeline.** Our voxel grids (``events.representations.voxel_grid``, E-RAFT convention)
    of the same event files + the vendored network (``models/eraft_ref``) + our warm-start runner;
    reported: max / mean |flow difference| to (a) and the EPE difference. Informational: float
    accumulation order differs, so it is not held to 1e-3 px.
(e) **Paper.** E-RAFT's own dense EPE (part b) within ``epe_tolerance_rel`` of the paper value.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from check_data import redact_home, sha256_file

from evflow.eval.datasets import load_yaml
from evflow.eval.run import eraft_voxel
from evflow.eval.windows import events_in_window, region_mask
from evflow.flow.metrics import flow_metrics
from evflow.models.eraft_runner import WarmStartRunner, load_eraft
from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "baselines" / "eraft_reproduction.json"
H, W = 260, 346


def read_events(path: Path) -> dict[str, np.ndarray]:
    """E-RAFT preprocessed event file: dataset ``myDataset`` (x i2, y i2, p i1 in {-1, +1},
    ts f8 Unix s). Times -> int64 µs (round), polarity -> {0, 1}."""
    with h5py.File(path, "r") as f:
        d = f["myDataset"][:]
    t = np.round(d["ts"].astype(np.float64) * 1e6).astype(np.int64)
    o = np.argsort(t, kind="stable")  # E-RAFT sorts by time if needed
    return {"x": d["x"][o].astype(np.int64), "y": d["y"][o].astype(np.int64), "t": t[o],
            "p": (d["p"][o] > 0).astype(np.int64)}


def our_masks(gt: np.ndarray, rep: dict, events_new: dict | None) -> np.ndarray:
    valid = (gt[0] != 0) | (gt[1] != 0)
    valid[int(rep["hood_rows_from"]):, :] = False
    if events_new is not None:
        valid &= events_in_window(events_new["x"], events_new["y"], H, W)
    return valid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, default=REPO / "runs/phase4-baselines/repro")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=OUT, help="trial runs: write elsewhere")
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ev_cfg = load_yaml("configs/eval_mvsec.yaml")
    rep = ev_cfg["reproduction_eraft"]
    er_cfg = load_yaml("configs/eraft.yaml")
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    root = Path(paths["eraft_mvsec_preprocessed"]) / "outdoor_day_1"
    ref_json = json.loads((args.reference / "reference.json").read_text())
    ref_npz = args.reference / "reference.npz"
    if sha256_file(ref_npz) != ref_json["reference_npz_sha256"]:
        raise SystemExit("reference.npz does not match reference.json")
    if ref_json["eraft_commit"] != er_cfg["source_commit"]:
        raise SystemExit(f"reference ran E-RAFT {ref_json['eraft_commit']}, "
                         f"vendored {er_cfg['source_commit']}")
    ref = dict(np.load(ref_npz))
    idx = ref["idx"]
    lo, hi = rep["indices"]
    if not np.array_equal(idx, np.arange(lo, hi)):
        raise SystemExit(f"reference samples {idx.min()}..{idx.max()} != range({lo}, {hi})")

    reg = ev_cfg["regions"][rep["crop"]]
    r0, r1 = reg["rows"]
    c0, c1 = reg["cols"]
    crop = torch.from_numpy(region_mask(reg, H, W))
    tol = float(rep["match_tolerance_px"])

    # (c) metric parity on E-RAFT's own flows; (d) our pipeline, run alongside
    arm = "eraft_mvsec20"
    ck = er_cfg["checkpoints"][arm]
    model = load_eraft(paths["checkpoints"][ck["key"]], sha256=ck["sha256"],
                       num_bins=er_cfg["num_bins"], device=args.device)
    runner = WarmStartRunner(model, iters=er_cfg["iters"])
    rows = {"dense": [], "sparse": []}
    mask_mismatch = {"dense": 0, "sparse": 0}
    flow_diff_max, flow_diff_mean, ours_pipeline = 0.0, [], {"dense": [], "sparse": []}
    prev_vox, prev_i = None, None
    for n, i in enumerate(idx):
        gt = np.load(root / "optical_flow" / f"{i:06d}.npy").astype(np.float64)
        ev_new = read_events(root / "davis/left/events" / f"{i + 1:06d}.h5")
        est_full = np.zeros((2, H, W))
        est_full[:, r0:r1, c0:c1] = ref["flow_est"][n]
        for kind in ("dense", "sparse"):
            valid = our_masks(gt, rep, ev_new if kind == "sparse" else None)
            ref_mask = ref[f"valid_{kind}"][n]
            mask_mismatch[kind] += int((valid[r0:r1, c0:c1] != ref_mask).sum())
            m = flow_metrics(torch.from_numpy(est_full), torch.from_numpy(gt),
                             valid_gt=torch.from_numpy(valid), crop=crop)
            rows[kind].append(m)

        # (d) our input pipeline
        if prev_i is None or i - prev_i != 1:
            runner.reset()
            prev_vox = eraft_voxel(read_events(root / "davis/left/events" / f"{i:06d}.h5"),
                                   bins=er_cfg["num_bins"], height=H, width=W,
                                   device=args.device)[:, r0:r1, c0:c1]
        vox_new = eraft_voxel(ev_new, bins=er_cfg["num_bins"], height=H, width=W,
                              device=args.device)[:, r0:r1, c0:c1]
        ours = runner(prev_vox, vox_new).double().cpu().numpy()
        prev_vox, prev_i = vox_new, i
        d = np.abs(ours - ref["flow_est"][n].astype(np.float64))
        flow_diff_max = max(flow_diff_max, float(d.max()))
        flow_diff_mean.append(float(d.mean()))
        ours_full = np.zeros((2, H, W))
        ours_full[:, r0:r1, c0:c1] = ours
        for kind in ("dense", "sparse"):
            valid = our_masks(gt, rep, ev_new if kind == "sparse" else None)
            ours_pipeline[kind].append(flow_metrics(
                torch.from_numpy(ours_full), torch.from_numpy(gt),
                valid_gt=torch.from_numpy(valid), crop=crop)["epe"])

    result = {}
    passed = True
    for kind in ("dense", "sparse"):
        ref_eval = ref_json[kind]
        ours_epe = np.array([m["epe"] for m in rows[kind]])
        ref_epe = np.array(ref_eval["per_sample_epe"])
        per_sample = float(np.max(np.abs(ours_epe - ref_epe)))
        mean_ours = float(ours_epe.mean())
        npe = {k: float(np.mean([m[k] for m in rows[kind]])) for k in ("npe_1", "npe_3")}
        d_mean = abs(mean_ours - ref_eval["epe"])
        d_npe = max(abs(npe["npe_1"] - ref_eval["npe_1"]), abs(npe["npe_3"] - ref_eval["npe_3"]))
        ok = (per_sample <= tol and d_mean <= tol and d_npe <= tol
              and mask_mismatch[kind] == 0)
        passed &= ok
        result[kind] = {
            "eraft_own_evaluation": {k: ref_eval[k] for k in
                                     ("epe", "npe_1", "npe_3", "epe_pooled", "n_samples")},
            "our_metrics_on_eraft_flows": {"epe": mean_ours, **npe},
            "max_abs_diff_per_sample_epe_px": per_sample,
            "abs_diff_mean_epe_px": d_mean,
            "max_abs_diff_npe_pct": d_npe,
            "mask_pixels_differing_from_eraft": mask_mismatch[kind],
            "metric_parity_pass": ok,
            "our_pipeline_epe": float(np.mean(ours_pipeline[kind])),
            "our_pipeline_minus_eraft_epe_px": float(np.mean(ours_pipeline[kind])
                                                     - ref_eval["epe"]),
        }

    paper = rep["paper"]
    target = float(paper["dense"]["epe"])
    got = float(ref_json["dense"]["epe"])
    rel = (got - target) / target
    paper_ok = abs(rel) <= float(paper["epe_tolerance_rel"])
    sha, dirty = git_sha(), git_dirty()
    doc = {
        "tier": "T1",
        "arm": arm,
        "model": "R1",
        "seed": None,
        "checkpoint_sha256": ck["sha256"],
        "dataset": "mvsec",
        "sequence": "outdoor_day1",
        "window_ms": None,
        "window": "E-RAFT 20 Hz: [timestamps_depth[i], timestamps_depth[i+1]], i in range(4356, "
                  "4706) (configs/eval_mvsec.yaml reproduction_eraft)",
        "representation": f"voxel{er_cfg['num_bins']} (E-RAFT convention)",
        "tags": ["reference_external", "reproduction_gate"],
        "metrics": {"epe": got, "npe_1": ref_json["dense"]["npe_1"],
                    "npe_3": ref_json["dense"]["npe_3"],
                    "epe_sparse": ref_json["sparse"]["epe"]},
        "metrics_source": "E-RAFT's own pipeline + evaluation (scripts/eraft_reference_run.py), "
                          "dense mask, per-sample mean then mean over samples",
        "ci95": None,
        "n_windows": int(len(idx)),
        "parity": result,
        "our_pipeline_flow_diff_px": {"max_abs": flow_diff_max,
                                      "mean_abs": float(np.mean(flow_diff_mean))},
        "paper": {"source": paper["source"], "dense_epe": target,
                  "sparse_epe": float(paper["sparse"]["epe"]), "rel_diff_dense_epe": rel,
                  "tolerance_rel": float(paper["epe_tolerance_rel"]), "within_tolerance": paper_ok},
        "gate": {"metric_parity_1e-3px": bool(passed), "epe_within_10pct_of_paper": paper_ok,
                 "pass": bool(passed and paper_ok)},
        "reference_run": {"json": redact_home(args.reference / "reference.json"),
                          "json_sha256": sha256_file(args.reference / "reference.json"),
                          "npz_sha256": ref_json["reference_npz_sha256"],
                          "eraft_commit": ref_json["eraft_commit"],
                          "resets_logged": ref_json["resets_logged"],
                          "wall_time_s": ref_json["wall_time_s"],
                          "torch": ref_json["torch"], "device": ref_json["device"]},
        "eraft_venv_lock": "configs/eraft_requirements.lock",
        "git_sha": sha,
        "git_dirty": dirty,
        "created_by": "scripts/eraft_reproduce.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"reproduction: E-RAFT dense EPE {got:.4f} vs paper {target} ({100 * rel:+.1f} %), "
          f"metric parity {'PASS' if passed else 'FAIL'} "
          f"(max per-sample {result['dense']['max_abs_diff_per_sample_epe_px']:.2e} px), "
          f"our pipeline dense EPE {result['dense']['our_pipeline_epe']:.4f}, "
          f"max |dflow| {flow_diff_max:.3e} px -> {redact_home(args.out)}")
    return 0 if doc["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
