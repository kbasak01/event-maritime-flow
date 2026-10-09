"""scripts/freeze_profile.py renders the calibrated values into the real profiles YAML."""

import importlib.util
import sys
from pathlib import Path

import yaml

from evflow.sim import v2e_runner

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("freeze_profile", REPO / "scripts/freeze_profile.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

DOC = {"argmin_params": {"cell": "c09", "pos_neg_thres": 0.2, "sigma_thres": 0.02,
                         "cutoff_hz": 60.0, "shot_noise_rate_hz": 1.0},
       "rand": {"thres_range": [0.14, 0.3], "sigma_range": [0.014, 0.04]},
       "ncc": {"threshold": 0.1234567}}


def test_render_and_idempotent():
    text = (REPO / "configs/v2e_profiles.yaml").read_text()
    vals = fp.frozen_values(DOC)
    out = fp.render(text, vals, "ab" * 32)
    assert fp.render(out, vals, "ab" * 32) == out
    doc = yaml.safe_load(out)
    assert doc["calibration_sha256"] == "ab" * 32
    assert doc["alignment_ncc_threshold"] == 0.123457
    rnd = doc["profiles"]["mvsec_rand"]["randomise"]
    assert rnd["thres_range"] == [0.14, 0.3] and rnd["sigma_range"] == [0.014, 0.04]
    args = v2e_runner.resolve_profile(doc, "mvsec_cal")["args"]
    assert args["--pos_thres"] == args["--neg_thres"] == "0.2"
    assert args["--cutoff_hz"] == "60.0" and args["--shot_noise_rate_hz"] == "1.0"
    # children of mvsec_cal inherit the calibrated pixel model
    assert v2e_runner.resolve_profile(doc, "maritime_640x360")["args"]["--sigma_thres"] == "0.02"
    # comments outside the rendered keys survive
    assert "# v2e argument profiles" in out
