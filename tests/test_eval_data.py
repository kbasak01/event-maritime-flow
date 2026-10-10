"""Evaluation units on the real data (marker ``dataset``): window counts match the split
manifests, GT and events are readable for the first and last window, masks are non-empty."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from evflow.eval.datasets import dsec_unit, load_yaml, mvsec_unit

pytestmark = pytest.mark.dataset
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def paths():
    p = REPO / "configs/paths.local.yaml"
    if not p.exists():
        pytest.skip("configs/paths.local.yaml missing")
    return yaml.safe_load(p.read_text())


@pytest.mark.parametrize("seq", ["indoor_flying1", "indoor_flying2", "indoor_flying3",
                                 "outdoor_day1"])
@pytest.mark.parametrize("dt", [1, 4])
def test_mvsec_test_units(paths, seq, dt):
    cfg = load_yaml("configs/eval_mvsec.yaml")
    region = cfg["regions"][cfg["sequences"][seq]["region"]]
    windows, data, meta = mvsec_unit(paths, seq, dt, split="mvsec_test", region=region)
    if seq == "outdoor_day1":
        assert len(windows) == cfg["sequences"][seq]["windows"]["expected_n_windows"][f"dt{dt}"]
    for w in (windows[0], windows[-1]):
        ev = data.slicer.get_events(w.t_start_us, w.t_end_us)
        flow, valid = data.gt(w)
        assert ev is not None and len(ev["t"]) > 0
        assert flow.shape == (2, 260, 346) and valid.any()
        assert w.prev_start_us is not None


def test_mvsec_tune_unit(paths):
    cfg = load_yaml("configs/eval_mvsec.yaml")
    windows, _, meta = mvsec_unit(paths, "outdoor_day2", 1, split="mvsec_tune",
                                  region=cfg["regions"]["center_256_no_hood"])
    assert len(windows) == 2624 and meta["split"] == "mvsec_tune"


@pytest.mark.parametrize("seq,n", [("thun_00_a", 41), ("zurich_city_02_a", 64),
                                   ("zurich_city_11_a", 231)])
def test_dsec_test_units(paths, seq, n):
    windows, data, _ = dsec_unit(paths, seq, delta_t_us=100_000)
    assert len(windows) == n
    assert all(w.prev_start_us is not None for w in windows[1:])
    w = windows[0]
    ev = data.slicer.get_events(w.t_start_us - 100_000, w.t_start_us)
    assert ev is not None and np.issubdtype(ev["x"].dtype, np.floating)
    flow, valid = data.gt(w)
    assert flow.shape == (2, 480, 640) and valid.any()
