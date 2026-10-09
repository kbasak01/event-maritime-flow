"""scripts/make_splits.py rules on synthetic inputs (no data): tune draw, gap rule, LaRS clips."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("make_splits", REPO / "scripts/make_splits.py")
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

S = 1_000_000


def test_tune_draw_deterministic_and_inside_span():
    ts = np.arange(0, 600 * S + 1, 50_000, dtype=np.int64) + 1_500_000_000 * S
    never = lambda a, b: (False, 1.0)  # noqa: E731
    t0, log = ms.draw_tune_slice(ts, 20261008, never)
    t0b, _ = ms.draw_tune_slice(ts, 20261008, never)
    assert t0 == t0b and len(log) == 1
    assert ts[0] <= t0 and t0 + ms.TUNE_LEN_US <= ts[-1]
    assert (t0 - ts[0]) % ms.TUNE_GRID_US == 0


def test_tune_draw_rejects_static():
    ts = np.arange(0, 600 * S + 1, 50_000, dtype=np.int64)
    calls = []

    def static_first(a, b):
        calls.append(a)
        return (len(calls) == 1, 0.0 if len(calls) == 1 else 2.0)

    t0, log = ms.draw_tune_slice(ts, 1, static_first)
    assert len(log) == 2 and log[0]["rejected_static"] and not log[1]["rejected_static"]
    assert t0 == log[1]["t_start_us"]


def test_train_ranges_gap_and_no_overlap():
    g0, g1 = 0, 600 * S
    tune0 = 200 * S
    tr = ms.train_ranges(g0, g1, tune0)
    assert tr == [[0, 170 * S], [290 * S, 600 * S]]
    tune1 = tune0 + ms.TUNE_LEN_US
    for a, b in tr:
        assert b <= tune0 - ms.TUNE_GAP_US or a >= tune1 + ms.TUNE_GAP_US


def test_train_ranges_tune_at_edges():
    assert ms.train_ranges(0, 600 * S, 0) == [[90 * S, 600 * S]]
    assert ms.train_ranges(0, 600 * S, 540 * S) == [[0, 510 * S]]


def test_aps_frames_in_half_open():
    aps = np.arange(10, dtype=np.int64) * 100
    r = ms.aps_frames_in(aps, 150, 500)
    assert (r["aps_first"], r["aps_last"], r["n_frames"], r["n_windows_dt1"]) == (2, 4, 3, 2)


def test_finalize_sha_stable_and_key_order_free():
    a = ms.finalize({"b": 1, "a": [1, 2]})
    b = ms.finalize({"a": [1, 2], "b": 1})
    assert a["manifest_sha256"] == b["manifest_sha256"]


@pytest.fixture
def lars_dir(tmp_path):
    d = tmp_path / "val"
    (d / "images").mkdir(parents=True)
    (d / "images_seq").mkdir()
    for pre, rng_, keys in [("seqA", range(0, 15), [9]), ("seqA", range(30, 40), [39]),
                            ("seqB", range(5, 20), [14])]:
        for n in rng_:
            (d / "images_seq" / f"{pre}_{n:05d}.jpg").write_bytes(b"x")
        for k in keys:
            (d / "images" / f"{pre}_{k:05d}.jpg").write_bytes(b"x")
    return d


def test_lars_clips_runs_and_keyframes(lars_dir):
    clips = ms.lars_clips(lars_dir)
    assert [(c["prefix"], c["first"], c["last"], c["n_frames"]) for c in clips] == [
        ("seqA", 0, 14, 15), ("seqA", 30, 39, 10), ("seqB", 5, 19, 15)]
    assert [c["keyframes"] for c in clips] == [["seqA_00009"], ["seqA_00039"], ["seqB_00014"]]
