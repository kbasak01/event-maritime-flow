"""Canonical event store and EventSlicer (IMPLEMENTATION_PLAN §1.2, DSEC ms_to_idx semantics)."""

import h5py
import numpy as np
import pytest

from evflow.io.store import (
    EventSlicer,
    EventStore,
    StoreWriter,
    compute_ms_to_idx,
    write_store,
)

META = dict(width=8, height=6, source="unit-test", sensor="synthetic", distortion="raw")

# DSEC eventslicer.py docstring (uzh-rpg/DSEC scripts/utils/eventslicer.py):
DSEC_T = np.array([0, 500, 2100, 5000, 5000, 7100, 7200, 7200, 8100, 9000], dtype=np.int64)
DSEC_MS_TO_IDX = [0, 2, 2, 3, 3, 3, 5, 5, 8, 9]


def _events(t):
    n = len(t)
    return dict(
        x=np.arange(n, dtype=np.uint16) % 8,
        y=np.arange(n, dtype=np.uint16) % 6,
        t=np.asarray(t, dtype=np.int64),
        p=(np.arange(n) % 2).astype(np.int8),
    )


def test_ms_to_idx_dsec_docstring_example():
    m = compute_ms_to_idx(DSEC_T)
    assert m.dtype == np.int64
    assert m[: len(DSEC_MS_TO_IDX)].tolist() == DSEC_MS_TO_IDX
    # DSEC files carry one entry past the last event's ms, pointing at N (checked on thun_00_a).
    assert len(m) == DSEC_T[-1] // 1000 + 2
    assert m[-1] == len(DSEC_T)


def test_ms_to_idx_definition_on_random_times():
    rng = np.random.default_rng(1)
    t = np.sort(rng.integers(0, 50_000, 5_000)).astype(np.int64)
    m = compute_ms_to_idx(t)
    for ms in range(len(m)):
        i = m[ms]
        if i < len(t):
            assert t[i] >= ms * 1000
        if i > 0:
            assert t[i - 1] < ms * 1000
    assert m[0] == 0


def test_ms_to_idx_rejects_decreasing_time():
    with pytest.raises(ValueError, match="non-decreasing"):
        compute_ms_to_idx(np.array([0, 10, 5], dtype=np.int64))


def test_round_trip_and_attrs(tmp_path):
    path = tmp_path / "s.h5"
    ev = _events(DSEC_T)
    write_store(path, **ev, t_offset=1_000_000, **META)
    with EventStore(path) as s:
        assert s.n_events == len(DSEC_T)
        assert s.t_offset == 1_000_000
        assert (s.width, s.height) == (8, 6)
        assert s.attrs["polarity_convention"] == "0=OFF,1=ON"
        assert s.attrs["distortion"] == "raw"
        assert s.ms_to_idx[: len(DSEC_MS_TO_IDX)].tolist() == DSEC_MS_TO_IDX
        assert not s.has_rectify_map
    with h5py.File(path, "r") as f:
        assert f["events/x"].dtype == np.uint16 and f["events/y"].dtype == np.uint16
        assert f["events/t"].dtype == np.int64 and f["events/p"].dtype == np.int8
        assert f["ms_to_idx"].dtype == np.int64 and f["t_offset"].dtype == np.int64
        np.testing.assert_array_equal(f["events/t"][:], DSEC_T)


def test_chunked_append_equals_single_write(tmp_path):
    rng = np.random.default_rng(2)
    t = np.sort(rng.integers(0, 37_000, 10_001)).astype(np.int64)
    ev = _events(t)
    write_store(tmp_path / "a.h5", **ev, t_offset=0, **META)
    with StoreWriter(tmp_path / "b.h5", t_offset=0, **META) as w:
        for lo in range(0, len(t), 997):
            w.append(**{k: v[lo : lo + 997] for k, v in ev.items()})
    with EventStore(tmp_path / "a.h5") as a, EventStore(tmp_path / "b.h5") as b:
        np.testing.assert_array_equal(a.ms_to_idx, b.ms_to_idx)
        np.testing.assert_array_equal(a.ms_to_idx, compute_ms_to_idx(t))


@pytest.mark.parametrize(
    "bad, msg",
    [
        (dict(p=np.array([0, 1, 2], dtype=np.int8)), "polarity"),
        (dict(x=np.array([0, 8, 1], dtype=np.uint16)), "x out of range"),
        (dict(y=np.array([0, 6, 1], dtype=np.uint16)), "y out of range"),
        (dict(t=np.array([0, 5, 3], dtype=np.int64)), "non-decreasing"),
        (dict(t=np.array([-1, 5, 6], dtype=np.int64)), "negative"),
    ],
)
def test_writer_rejects_invalid(tmp_path, bad, msg):
    ev = _events([0, 1, 2])
    ev.update(bad)
    with pytest.raises(ValueError, match=msg):
        write_store(tmp_path / "bad.h5", **ev, t_offset=0, **META)


def test_writer_rejects_non_monotone_across_chunks(tmp_path):
    with pytest.raises(ValueError, match="non-decreasing"):
        with StoreWriter(tmp_path / "c.h5", t_offset=0, **META) as w:
            w.append(**_events([0, 10, 20]))
            w.append(**_events([19, 30]))


def test_writer_requires_attrs(tmp_path):
    with pytest.raises(ValueError, match="distortion"):
        write_store(tmp_path / "d.h5", **_events([0]), t_offset=0,
                    width=8, height=6, source="s", sensor="s", distortion="weird")


# --- slicer ----------------------------------------------------------------------------------


@pytest.fixture
def dsec_like(tmp_path):
    path = tmp_path / "s.h5"
    write_store(path, **_events(DSEC_T), t_offset=100_000, **META)
    return path


def test_slicer_half_open_window(dsec_like):
    with EventSlicer(dsec_like) as sl:
        ev = sl.get_events(100_000 + 500, 100_000 + 5000)  # [500, 5000)
        assert ev["t"].tolist() == [100_500, 102_100]       # absolute time, like DSEC
        ev = sl.get_events(100_000 + 500, 100_000 + 5001)
        assert ev["t"].tolist() == [100_500, 102_100, 105_000, 105_000]
        assert ev["x"].dtype == np.uint16 and ev["p"].dtype == np.int8


def test_slicer_matches_bruteforce(tmp_path):
    rng = np.random.default_rng(3)
    t = np.sort(rng.integers(0, 40_000, 4_000)).astype(np.int64)
    write_store(tmp_path / "r.h5", **_events(t), t_offset=7, **META)
    with EventSlicer(tmp_path / "r.h5") as sl:
        for _ in range(200):
            a = int(rng.integers(0, 39_000))
            b = a + int(rng.integers(1, 2_000))
            ev = sl.get_events(a + 7, b + 7)
            want = t[(t >= a) & (t < b)] + 7
            np.testing.assert_array_equal(ev["t"], want)


def test_slicer_empty_window(dsec_like):
    with EventSlicer(dsec_like) as sl:
        ev = sl.get_events(100_000 + 2200, 100_000 + 4900)
        assert ev is not None and all(len(v) == 0 for v in ev.values())


def test_slicer_past_end_and_before_start_return_none(dsec_like):
    with EventSlicer(dsec_like) as sl:
        assert sl.get_events(100_000 + 9000, 100_000 + 20_000) is None
        assert sl.get_events(100_000 - 10, 100_000 + 100) is None
        # last retrievable window ends at the ms after the last event (DSEC ms_to_idx length)
        ev = sl.get_events(100_000 + 8000, 100_000 + 10_000)
        assert ev["t"].tolist() == [108_100, 109_000]
        with pytest.raises(ValueError):
            sl.get_events(100_000 + 50, 100_000 + 50)


def test_slicer_rectify_on_read(tmp_path):
    rmap = np.zeros((6, 8, 2), dtype=np.float32)
    ys, xs = np.mgrid[0:6, 0:8]
    rmap[..., 0] = xs + 0.25
    rmap[..., 1] = ys - 0.5
    path = tmp_path / "r.h5"
    write_store(path, **_events(DSEC_T), t_offset=0, rectify_map=rmap, **META)
    with EventSlicer(path) as sl:
        raw = sl.get_events(0, 10_000, rectify=False)
        assert sl.get_events(0, 10_000)["x"].dtype == np.float32  # default: rectify if map
        rect = sl.get_events(0, 10_000, rectify=True)
    assert rect["x"].dtype == np.float32 and rect["y"].dtype == np.float32
    np.testing.assert_allclose(rect["x"], raw["x"] + 0.25)
    np.testing.assert_allclose(rect["y"], raw["y"] - 0.5)
    np.testing.assert_array_equal(rect["t"], raw["t"])


def test_rectify_without_map_raises(dsec_like):
    with EventSlicer(dsec_like) as sl, pytest.raises(ValueError, match="rectify_map"):
        sl.get_events(0 + 100_000, 1000 + 100_000, rectify=True)


def test_rectify_map_shape_checked(tmp_path):
    with pytest.raises(ValueError, match="rectify_map"):
        write_store(tmp_path / "x.h5", **_events([0]), t_offset=0,
                    rectify_map=np.zeros((6, 7, 2), np.float32), **META)
