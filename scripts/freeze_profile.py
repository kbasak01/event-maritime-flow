"""Freeze the calibrated v2e profile from results/v2e_calibration.json (Phase 2 exit gate).

    python scripts/freeze_profile.py [--check]

Re-derives the argmin from the stored histograms (must equal the JSON's), then writes into
configs/v2e_profiles.yaml, preserving comments: ``mvsec_cal.args`` (the argmin's calibrated flags),
``mvsec_rand.randomise.{thres_range,sigma_range}``, ``alignment_ncc_threshold`` and
``calibration_sha256`` (SHA-256 of the calibration JSON). ``--check`` verifies instead of writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

from evflow.sim import v2e_runner
from evflow.sim.calibrate import RESULT, cell_overrides, recompute_argmin

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "configs/v2e_profiles.yaml"


def frozen_values(doc: dict) -> dict:
    am = doc["argmin_params"]
    return {"args": cell_overrides(am), "thres_range": doc["rand"]["thres_range"],
            "sigma_range": doc["rand"]["sigma_range"],
            "ncc_threshold": round(float(doc["ncc"]["threshold"]), 6)}


def render(text: str, vals: dict, sha: str) -> str:
    args = ", ".join(f'{k}: "{v}"' for k, v in vals["args"].items())
    subs = [
        (r"(\n  mvsec_cal:[^\n]*\n    base: mvsec_grid\n    args: )[^\n]*", rf"\g<1>{{{args}}}"),
        (r"(\n      thres_range: )\[[^\]]*\]", rf"\g<1>{json.dumps(vals['thres_range'])}"),
        (r"(\n      sigma_range: )\[[^\]]*\]", rf"\g<1>{json.dumps(vals['sigma_range'])}"),
        (r"(\nalignment_ncc_threshold: )\S+", rf"\g<1>{vals['ncc_threshold']}"),
        (r"(\ncalibration_sha256: )\S+", rf"\g<1>{sha}"),
    ]
    for pat, rep in subs:
        text, n = re.subn(pat, rep, text, count=1)
        if n != 1:
            raise RuntimeError(f"pattern not found in {PROFILES.name}: {pat}")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    raw = RESULT.read_bytes()
    doc = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    if recompute_argmin(doc) != doc["argmin"]:
        print("argmin does not reproduce from the stored histograms", file=sys.stderr)
        return 1
    vals = frozen_values(doc)
    text = PROFILES.read_text()
    new = render(text, vals, sha)
    if a.check:
        ok = new == text
        prof = yaml.safe_load(text)
        merged = v2e_runner.resolve_profile(prof, "mvsec_cal")
        ok &= all(merged["args"][k] == v for k, v in vals["args"].items())
        print("profile frozen and consistent" if ok else "profile NOT consistent with JSON")
        return 0 if ok else 1
    PROFILES.write_text(new)
    print(json.dumps({"calibration_sha256": sha, **vals}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
