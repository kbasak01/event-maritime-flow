"""Export the selected M1 checkpoints to ONNX (Phase 8 step 1; D-063).

    python scripts/export_onnx.py                 # runs and shapes from configs/evflow.yaml
    python scripts/export_onnx.py --smoke --out-dir <scratch>

For each run in ``export.runs`` (checkpoint = the run's ``best.pt`` per ``selection.json``,
selected on ``mvsec_tune``, SHA-256 verified by ``run_eval.load_run``): opset 17, fp32 and fp16,
static (1, C, H, W) at the parity resolution 272x352 (346x260 padded to /16); the primary run
(first) also at every ``latency.resolutions`` entry padded to /16, for the latency benchmark.
Writes ``<out>/<run_id>/m1_<H>x<W>_<fp32|fp16>.onnx`` + sidecar JSON (gitignored ``runs/``).
Skips a file whose ONNX and sidecar exist unless ``--force``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from run_eval import load_run

from evflow.eval.datasets import load_yaml
from evflow.export.onnx import export_m1, padded_hw
from evflow.io import mvsec as mvsec_io
from evflow.sim.v2e_runner import git_dirty, git_sha
from evflow.train.data import in_channels

REPO = Path(__file__).resolve().parents[1]


def shapes_for(cfg: dict, primary: bool) -> list[tuple[int, int]]:
    mult = int(cfg["export"]["pad_multiple"])
    parity = padded_hw(mvsec_io.HEIGHT, mvsec_io.WIDTH, mult)
    if not primary:
        return [parity]
    out = [padded_hw(h, w, mult) for h, w in cfg["latency"]["resolutions"]]
    return out if parity in out else [*out, parity]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", help="default: <outputs.runs>/export")
    ap.add_argument("--runs", nargs="*", help="override export.runs")
    ap.add_argument("--smoke", action="store_true", help="parity shape only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_yaml("configs/evflow.yaml")
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    runs_root = Path(paths["outputs"]["runs"])
    out_root = Path(args.out_dir) if args.out_dir else runs_root / "export"
    run_ids = args.runs or cfg["export"]["runs"]
    sha, dirty = git_sha(), git_dirty()
    for i, run_id in enumerate(run_ids):
        model, sel, man = load_run(runs_root / "train" / run_id, args.device)
        cin = in_channels(man["representation"])
        shapes = shapes_for(cfg, primary=(i == 0 and not args.smoke))
        for h, w in shapes:
            for fp16 in (False, True):
                path = out_root / run_id / f"m1_{h}x{w}_{'fp16' if fp16 else 'fp32'}.onnx"
                if path.exists() and path.with_suffix(".json").exists() and not args.force:
                    print(f"skip: {path} exists")
                    continue
                meta = {"run_id": run_id, "arm": man["arm"], "seed": man["seed"],
                        "representation": man["representation"],
                        "checkpoint": sel["checkpoint"],
                        "checkpoint_sha256": sel["checkpoint_sha256"],
                        "selection": {k: sel[k] for k in ("rule", "split", "stream", "metric",
                                                          "dt", "selected_epoch", "value")},
                        "git_sha": sha, "git_dirty": dirty,
                        "created_by": "scripts/export_onnx.py"}
                info = export_m1(model, path, in_channels=cin, height=h, width=w, fp16=fp16,
                                 opset=int(cfg["export"]["opset"]), meta=meta)
                print(f"{run_id} {h}x{w} {info['precision']}: {info['n_nodes']} nodes, "
                      f"ops {info['op_types']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
