"""Phase 5 training loop on CPU with synthetic moving-dot windows: learning, selection record,
resume, the CM path, and the selection-split guard."""

import copy
import csv
import json

import pytest
import torch
from test_train_data import H, W, synthetic

from evflow.train.data import Augment
from evflow.train.loop import load_config, run_training, sha256_file, validate


def _cfg(loss="supervised", epochs=2):
    cfg = load_config("configs/train_real_sup.yaml")
    cfg = copy.deepcopy({k: v for k, v in cfg.items() if k != "_files"}) | {"_files": cfg["_files"]}
    cfg["seed"] = 0
    cfg["loss"] = loss
    t = cfg["train"]
    t.update(epochs=epochs, batch_size=4, precision="fp32", representation="voxel5",
             epoch_samples=8, dt_fractions={1: 0.5, 2: 0.5}, tune_diag_windows=3,
             tune_region={"rows": [0, H], "cols": [0, W]}, lr=3e-3)
    cfg["model_m1"]["base_channels"] = 4
    cfg["losses"]["cm"]["max_events"] = 5000
    if loss == "cm":
        cfg["selection"] = {"split": "mvsec_tune", "metric": "fwl", "mode": "max", "dt": 1}
    return cfg


def _data():
    train, _ = synthetic(augment=Augment(crop=(32, 48), hflip=True), hood_row=40)
    tune, _ = synthetic(seed=7)
    return train, {"real": tune}


def _rows(run_dir):
    with open(run_dir / "metrics.csv") as f:
        return list(csv.DictReader(f))


def test_train_select_resume(tmp_path):
    torch.manual_seed(0)
    train, tune = _data()
    run_dir = tmp_path / "run"
    sel = run_training(_cfg(epochs=2), run_dir, train, tune, device="cpu", log=lambda s: None)
    assert (run_dir / "DONE").exists()
    rows = _rows(run_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1]
    epes = [float(r["real_dt1_epe"]) for r in rows]
    assert sel["selected_epoch"] == int(min(range(2), key=lambda i: epes[i]))
    assert sel["checkpoint_sha256"] == sha256_file(run_dir / "best.pt")
    assert sel["split"] == "mvsec_tune"
    man = json.loads((run_dir / "manifest.json").read_text())
    for k in ("git_sha", "seed", "config_files", "train_windows", "selection", "n_params"):
        assert k in man
    assert "real_dt2_epe" in rows[0] and "real_dt1_diag_iwe_nonempty_ratio" in rows[0]

    # a finished run is skipped
    assert run_training(_cfg(epochs=2), run_dir, train, tune, device="cpu",
                        log=lambda s: None) == sel
    # extend to 3 epochs: resumes from last.pt at epoch 2, keeps the earlier rows
    (run_dir / "DONE").unlink()
    logs = []
    run_training(_cfg(epochs=3), run_dir, train, tune, device="cpu", log=logs.append)
    assert any("resuming at epoch 2" in s for s in logs)
    assert [int(r["epoch"]) for r in _rows(run_dir)] == [0, 1, 2]


def test_supervised_training_reduces_tune_epe(tmp_path):
    torch.manual_seed(0)
    train, tune = _data()
    cfg = _cfg(epochs=6)
    cfg["train"]["epoch_samples"] = 16
    run_training(cfg, tmp_path / "r", train, tune, device="cpu", log=lambda s: None)
    epes = [float(r["real_dt1_epe"]) for r in _rows(tmp_path / "r")]
    assert min(epes[1:]) < epes[0]


def test_cm_arm_runs_and_selects_on_fwl(tmp_path):
    torch.manual_seed(0)
    train, tune = _data()
    sel = run_training(_cfg(loss="cm", epochs=1), tmp_path / "cm", train, tune, device="cpu",
                       log=lambda s: None)
    assert sel["metric"] == "fwl" and sel["mode"] == "max"
    row = _rows(tmp_path / "cm")[0]
    assert "cm_data" in row and "cm_div" in row and "train_epe" in row


def test_selection_split_guard():
    cfg = _cfg()
    cfg["selection"]["split"] = "mvsec_test"
    with pytest.raises(ValueError):
        validate(cfg)
