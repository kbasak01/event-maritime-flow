"""fp16 sensitivity of the exported M1 checkpoints, on mvsec_tune windows (Phase 8; D-064).

    python scripts/fp16_sensitivity.py            # -> results/fp16_sensitivity.json

Why fp16 parity fails (Phase 8 gate F-1) and whether a mixed-precision export can pass it,
measured on **tune** data only (CLAUDE.md rule 4: no precision choice is made on mvsec_test):
200 evenly spaced ``mvsec_tune`` outdoor_day2 dt1 windows, inputs captured from
``run_eval.NetPredictor``, reference = torch fp32 (TF32 off). Variants, each compared with the
reference by max / p99 over all pixels of |Δflow| (vector, px, unpadded 260x346):
- isolating perturbations in fp32 compute: input rounded to fp16; weights (+ BN buffers)
  rounded to fp16;
- fp16 network (``.half()``); bf16 autocast (the training precision);
- mixed precision: listed modules run in fp32 (weights and compute) inside an fp16 network; the
  tensors passed between modules stay fp16. ``all_modules_fp32`` keeps every module in fp32, so
  only the inter-module activations are fp16.
Diagnostic only: no number here is gated and no checkpoint or window is chosen from it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from run_eval import NetPredictor, load_run

from evflow.eval.datasets import mvsec_unit
from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "fp16_sensitivity.json"
ENC = ["enc.0", "enc.1", "enc.2", "enc.3"]
DEC_H = ["dec.0", "dec.1", "dec.2", "dec.3", "heads.0", "heads.1", "heads.2", "heads.3"]
MIXED = {
    "enc01": ["enc.0", "enc.1"],
    "enc012": ["enc.0", "enc.1", "enc.2"],
    "enc": ENC,
    "enc+res": [*ENC, "res"],
    "dec23+heads23": ["dec.2", "dec.3", "heads.2", "heads.3"],
    "enc01+dec23+heads23": ["enc.0", "enc.1", "dec.2", "dec.3", "heads.2", "heads.3"],
    "all_but_res": [*ENC, *DEC_H],
    "all_modules_fp32": [*ENC, "res", *DEC_H],
}


class _Keep32(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = copy.deepcopy(m).float()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.m(t.float()).half()


class _Capture(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net, self.last = net, None

    def forward(self, x):
        self.last = x.detach().clone()
        return self.net(x)


def _mixed(model: nn.Module, keep: list[str]) -> nn.Module:
    m = copy.deepcopy(model).half()
    for name in keep:
        if "." in name:
            parent, i = name.split(".")
            getattr(m, parent)[int(i)] = _Keep32(getattr(model, parent)[int(i)])
        else:
            setattr(m, name, _Keep32(getattr(model, name)))
    return m


def _rounded_weights(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    with torch.no_grad():
        for t in [*m.parameters(), *(b for b in m.buffers() if b.is_floating_point())]:
            t.copy_(t.half().float())
    return m


def _stats(y: torch.Tensor, ref: torch.Tensor) -> dict:
    d = ((y.float() - ref) ** 2).sum(1).sqrt().double().cpu().numpy()
    return {"max_px": float(d.max()), "p99_px": float(np.percentile(d, 99)),
            "mean_px": float(d.mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load((REPO / "configs/evflow.yaml").read_text())
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    windows, data, meta = mvsec_unit(paths, "outdoor_day2", 1, split="mvsec_tune",
                                     region={"rows": [0, 260], "cols": [0, 346]})
    if meta["split"] != "mvsec_tune":
        raise SystemExit("fp16 sensitivity runs on mvsec_tune only")
    pick = np.unique(np.linspace(0, len(windows) - 1, args.windows).round().astype(np.int64))
    h, w = data.height, data.width
    units = []
    for run_id in cfg["export"]["runs"]:
        model, sel, man = load_run(Path(paths["outputs"]["runs"]) / "train" / run_id, "cuda")
        cap = _Capture(model).eval()
        pred = NetPredictor(cap, man["representation"], h, w, "cuda")
        xs, refs = [], []
        for j in pick:
            win = windows[int(j)]
            refs.append(torch.from_numpy(pred(win, data.slicer.get_events(
                win.t_start_us, win.t_end_us))).float().cuda())
            xs.append(cap.last)
        x, ref = torch.cat(xs), torch.stack(refs)

        def run(net, dtype, inp=x):
            with torch.inference_mode():
                return torch.stack([net(inp[i: i + 1].to(dtype))[-1].float()[0, :, :h, :w]
                                    for i in range(inp.shape[0])])

        variants = {
            "input_rounded_fp16__fp32_compute": _stats(run(model, torch.float32,
                                                           x.half().float()), ref),
            "weights_rounded_fp16__fp32_compute": _stats(run(_rounded_weights(model),
                                                             torch.float32), ref),
            "fp16_network": _stats(run(copy.deepcopy(model).half(), torch.float16), ref),
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            variants["bf16_autocast"] = _stats(run(model, torch.float32), ref)
        for name, keep in MIXED.items():
            variants[f"mixed:{name}"] = {"fp32_modules": keep,
                                         **_stats(run(_mixed(model, keep), torch.float16),
                                                  ref)}
        units.append({"run_id": run_id, "arm": man["arm"], "seed": man["seed"],
                      "checkpoint_sha256": sel["checkpoint_sha256"], "variants": variants})
        print(run_id, {k: round(v["max_px"], 4) for k, v in variants.items()}, flush=True)
    doc = {
        "kind": "fp16_sensitivity",
        "plan_step": "Phase 8 gate F-1 follow-up (D-064)",
        "split": "mvsec_tune", "sequence": "outdoor_day2", "dt": 1,
        "split_manifest_sha256": meta["split_manifest_sha256"],
        "n_windows": int(pick.size),
        "window_rule": "np.unique(np.linspace(0, n - 1, 200).round()) over the tune unit",
        "reference": "torch fp32 eager, CUDA, TF32 off (run_eval.NetPredictor path)",
        "definition": "max / p99 / mean over all pixels of all windows of |flow_variant - "
                      "flow_fp32| (2-vector norm, px), unpadded 260x346 frame",
        "fp16_tolerance_px": cfg["export"]["parity"]["tolerances"]["fp16_max_px"],
        "note": "diagnostic, not gated; tune data only; nothing is chosen from mvsec_test",
        "tier_note": "T1 inputs (real mvsec_tune events); numbers are numerical deviations "
                     "from the fp32 prediction, not accuracy against GT",
        "units": units,
        "git_sha": git_sha(), "git_dirty": git_dirty(),
        "created_by": "scripts/fp16_sensitivity.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
