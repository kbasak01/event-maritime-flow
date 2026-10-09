"""A real v2e output vs its source (marker ``dataset``; run with ``make test-data``).

Needs the throughput unit written by jobs/2026-10-08_v2e-throughput.sh:
``<outputs.events>/v2e/throughput/mvsec_auto_ts500.{h5,json,v2e.json}`` (first 5 s of mvsec_tune).
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from evflow.io.mvsec import read_aps_timestamps_us
from evflow.io.store import EventStore, compute_ms_to_idx

pytestmark = pytest.mark.dataset

REPO = Path(__file__).resolve().parents[1]
UNIT = "mvsec_auto_ts500"


@pytest.fixture(scope="module")
def unit():
    p = REPO / "configs/paths.local.yaml"
    if not p.exists():
        pytest.skip("configs/paths.local.yaml missing")
    cfg = yaml.safe_load(p.read_text())
    d = Path(cfg["outputs"]["events"]) / "v2e" / "throughput"
    if not (d / f"{UNIT}.h5").exists():
        pytest.fail(f"{d / UNIT}.h5 missing: run jobs/2026-10-08_v2e-throughput.sh")
    return cfg, d


def test_manifest_links_and_policy(unit):
    _, d = unit
    m = json.loads((d / f"{UNIT}.json").read_text())
    rm_sha = hashlib.sha256((d / f"{UNIT}.v2e.json").read_bytes()).hexdigest()
    assert m["run_manifest_sha256"] == rm_sha
    with EventStore(d / f"{UNIT}.h5") as s:
        assert s.attrs["simulator_manifest_sha256"] == rm_sha
    assert hashlib.sha256((d / f"{UNIT}.h5").read_bytes()).hexdigest() == m["store_sha256"]
    argv = m["v2e_argv"]
    for f in ("--no_preview", "--skip_video_output", "--dvs_h5", "--dvs_emulator_seed",
              "--input_frame_rate", "--dvs346"):
        assert f in argv
    assert "--dvs_params" not in argv
    assert m["v2e_commit"] == (REPO / "configs/v2e_commit.txt").read_text().strip()
    assert m["warnings"]["undersampling"] == []


def test_store_integrity_and_clock(unit):
    cfg, d = unit
    m = json.loads((d / f"{UNIT}.json").read_text())
    aps = read_aps_timestamps_us(Path(cfg["mvsec"]["root"]) / "outdoor_day" /
                                 "outdoor_day2_data.hdf5")
    a, b = m["input"]["aps_first"], m["input"]["aps_last"]
    with EventStore(d / f"{UNIT}.h5") as s:
        t = s.events["t"][:]
        p = s.events["p"][:]
        assert np.all(np.diff(t) >= 0) and set(np.unique(p)) <= {0, 1}
        assert s.events["x"][:].max() < 346 and s.events["y"][:].max() < 260
        np.testing.assert_array_equal(s.ms_to_idx, compute_ms_to_idx(t))
        assert s.t_offset == aps[a]                                  # frame a = v2e time 0
        assert t[0] >= 0 and s.t_offset + t[-1] <= aps[b]             # within the frame span
        assert m["clock_map"]["uniform_vs_real_max_abs_dev_us"] < 100  # APS jitter is µs-level


def test_polarity_follows_log_intensity(unit):
    cfg, d = unit
    m = json.loads((d / f"{UNIT}.json").read_text())
    cache = Path(m["input"]["frames_dir"]).parent.name
    frames = Path(cfg["outputs"]["events"]) / "aps_cache" / cache / "frames"
    if not frames.is_dir():
        pytest.skip("APS frame cache missing")
    ts = np.loadtxt(frames.parent / "aps_timestamps_us.txt", dtype=np.int64)
    with EventStore(d / f"{UNIT}.h5") as s:
        t = s.events["t"][:] + s.t_offset
        x, y, p = s.events["x"][:], s.events["y"][:], s.events["p"][:]
    corrs = []
    for k in range(0, len(ts) - 4, 20):
        sel = (t >= ts[k]) & (t < ts[k + 4])
        net = np.zeros((260, 346))
        np.add.at(net, (y[sel], x[sel]), 2.0 * p[sel] - 1)
        i0 = cv2.imread(str(frames / f"{k:06d}.png"), 0).astype(float)
        i1 = cv2.imread(str(frames / f"{k + 4:06d}.png"), 0).astype(float)
        corrs.append(np.corrcoef(net.ravel(), (np.log1p(i1) - np.log1p(i0)).ravel())[0, 1])
    assert min(corrs) > 0.3, corrs
