"""Alignment check, real DVS vs v2e(APS) count images (Phase 3 step 3; checklist B4).

    python scripts/alignment_check.py [--allow-missing]

Per APS dt=1 window (owned, after warm-up; ``sim/fidelity.py``): NCC of the Gaussian-blurred
(σ from ``calibration_grid.ncc_rule``) count images of real and v2e events, plus negative
controls: v2e window 1 and 5 frames later (timing error) and v2e mirrored in x (geometry error).
Groups: tune, each test sequence, the train_cal clips (pooled and per clip), each randomised
corpus. The threshold is the one fixed on ``mvsec_tune`` in Phase 2
(``configs/v2e_profiles.yaml: alignment_ncc_threshold``, D-029); this script never sets it. The
Phase 3 gate requires the median NCC of every ``mvsec_cal`` group (tune, each test sequence,
train_cal) to exceed it (D-034). Writes ``results/v2e_alignment.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from fidelity_report import HEIGHT, WIDTH, read_real, read_syn

from evflow.io.mvsec import read_aps_timestamps_us
from evflow.sim import corpus as K
from evflow.sim import fidelity as F

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v2e_alignment.json"
CAL_GROUPS_PREFIX = ("tune_", "test_", "train_cal")
SCOPE = [
    "Detects: frame-timing or clock-map errors of one frame or more (shift controls), flipped "
    "or transposed geometry (mirror control), grossly wrong pixel mapping, and missing or "
    "displaced chunks.",
    "Does not detect: contrast-threshold or noise mismatch (count magnitudes cancel in the "
    "NCC), sub-frame timing errors much smaller than one APS interval, or v2e's timestamp "
    "pyramid within a frame interval.",
    "Scene dependence: the shift controls stay high when the scene moves slowly between frames "
    "(outdoor driving at low speed, D-029), so their margin below the aligned NCC is the "
    "evidence, not their absolute value; indoor flying scenes have sparser texture and faster "
    "rotation, so their NCC level is not comparable with outdoor_day2 in absolute terms.",
]


def group_of(u: dict) -> str | None:
    if u["corpus"] == "tune":
        return f"tune_{u['sequence']}"
    if u["corpus"] == "test":
        return f"test_{u['sequence']}"
    if u["corpus"] in ("train_cal",) or u["corpus"].startswith("rand"):
        return u["corpus"]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the JSON although planned units are missing (inspection only)")
    a = ap.parse_args(argv)
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    prof = yaml.safe_load((REPO / "configs/v2e_profiles.yaml").read_text())
    thr = float(prof["alignment_ncc_threshold"])
    sigma = float(prof["calibration_grid"]["ncc_rule"]["blur_sigma_px"])
    plan = K.load_plan(paths)
    if plan["units"] != K.build_plan()["units"]:
        raise SystemExit("corpus_plan.json differs from the plan rebuilt from configs + splits")
    root = K.out_root(paths)
    missing = [u["key"] for u in plan["units"] if group_of(u) is not None
               and not (root / u["corpus"] / f"{u['name']}.json").exists()]
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} planned units missing (e.g. {missing[:3]}); "
                         "nothing written")
    aps: dict[str, np.ndarray] = {}
    vals: dict[str, dict[str, list]] = {}
    per_unit = []
    for u in plan["units"]:
        g = group_of(u)
        if g is None:
            continue
        seq = u["sequence"]
        if seq not in aps:
            aps[seq] = read_aps_timestamps_us(K._data_file(paths, seq))
        a = aps[seq]
        syn_store = root / u["corpus"] / f"{u['name']}.h5"
        if u["key"] in missing:
            continue
        real = read_real(Path(paths["outputs"]["events"]) / f"mvsec_{seq}_left.h5",
                         int(a[u["aps"][0]]), int(a[u["aps"][1]]))
        n = F.unit_ncc(real, read_syn(syn_store), a, u, WIDTH, HEIGHT, sigma)
        for k, v in n.items():
            vals.setdefault(g, {}).setdefault(k, []).append(v)
        man = json.loads(syn_store.with_suffix(".json").read_text())
        per_unit.append({"key": u["key"], "group": g, "profile": u["profile"],
                         "v2e_store_sha256": man["store_sha256"],
                         **F.ncc_summary({k: [v] for k, v in n.items()}, thr)})
        print(json.dumps({"unit": u["key"], "median": per_unit[-1]["median"]}))
    groups = {g: {"profile": "mvsec_cal" if g.startswith(CAL_GROUPS_PREFIX) else "mvsec_rand",
                  "gate": g.startswith(CAL_GROUPS_PREFIX), **F.ncc_summary(v, thr)}
              for g, v in vals.items()}
    gate = {g: d["pass"] for g, d in groups.items() if d["gate"]}
    doc = {
        "kind": "v2e_alignment", "tier": "T2",
        "tier_note": "count-image co-location of v2e(MVSEC APS) and real DVS events; a "
                     "pipeline check, not flow accuracy",
        "plan_step": "Phase 3 step 3", "dataset": "mvsec", "used_for_choice": False,
        "threshold": thr, "threshold_source": "configs/v2e_profiles.yaml "
                      "alignment_ncc_threshold (mvsec_tune, Phase 2, D-029)",
        "calibration_sha256": prof["calibration_sha256"], "blur_sigma_px": sigma,
        "controls": {"shift1": "v2e window 1 APS frame later", "shift5": "v2e window 5 APS "
                     "frames later", "mirror": "v2e x mirrored (x -> W-1-x)"},
        "gate_rule": "median NCC > threshold for every mvsec_cal group (D-034)",
        "missing_units": missing,
        "gate": gate, "gate_pass": bool(gate) and all(gate.values()) and not missing,
        "groups": groups, "units": per_unit, "scope": SCOPE,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                  text=True).stdout.strip(),
        "created_by": "scripts/alignment_check.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"gate_pass": doc["gate_pass"], **{g: round(d["median"], 4)
                                                          for g, d in groups.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
