"""io/v2e_reader on fake files with the observed v2e layout (D-022): (N, 4) uint32 [t, x, y, p]."""

import h5py
import numpy as np
import pytest

from evflow.io.store import EventSlicer, EventStore, compute_ms_to_idx
from evflow.io.v2e_reader import convert_v2e_h5, read_v2e_events


def write_fake_v2e(path, rows):
    with h5py.File(path, "w") as f:
        f.create_dataset("events", data=np.asarray(rows, np.uint32), maxshape=(None, 4),
                         chunks=(512, 4), compression="gzip")


@pytest.fixture
def fake(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    t = np.sort(rng.integers(0, 2_000_000, n))
    rows = np.stack([t, rng.integers(0, 346, n), rng.integers(0, 260, n),
                     rng.integers(0, 2, n)], 1)
    src = tmp_path / "v2e.h5"
    write_fake_v2e(src, rows)
    return src, rows


def test_read_columns(fake):
    src, rows = fake
    ev = read_v2e_events(src)
    np.testing.assert_array_equal(ev["t"], rows[:, 0])
    np.testing.assert_array_equal(ev["x"], rows[:, 1])
    np.testing.assert_array_equal(ev["y"], rows[:, 2])
    np.testing.assert_array_equal(ev["p"], rows[:, 3])
    assert ev["t"].dtype == np.int64 and ev["p"].dtype == np.int8


def test_convert_identity_clock(fake, tmp_path):
    src, rows = fake
    dst = tmp_path / "s.h5"
    s = convert_v2e_h5(src, dst, width=346, height=260, source="v2e/test", sensor="v2e(test)",
                       t_offset_us=1_000_000_000, simulator_manifest_sha256="ab" * 32, chunk=777)
    assert s["n_events"] == len(rows) and s["n_on"] == rows[:, 3].sum()
    with EventStore(dst) as st:
        assert st.attrs["simulator_manifest_sha256"] == "ab" * 32
        assert st.attrs["distortion"] == "raw" and st.t_offset == 1_000_000_000
        t = st.events["t"][:]
        np.testing.assert_array_equal(t, rows[:, 0])
        np.testing.assert_array_equal(st.ms_to_idx, compute_ms_to_idx(t))
    with EventSlicer(dst) as sl:
        w = sl.get_events(1_000_500_000, 1_001_000_000)
        m = (rows[:, 0] >= 500_000) & (rows[:, 0] < 1_000_000)
        np.testing.assert_array_equal(w["x"], rows[m, 1])


def test_convert_with_time_map_and_sort(tmp_path):
    # map that swaps order of two events -> stable sort keeps the x/y/p with their times
    rows = np.array([[0, 1, 1, 1], [10, 2, 2, 0], [20, 3, 3, 1]])
    src = tmp_path / "v.h5"
    write_fake_v2e(src, rows)
    dst = tmp_path / "s.h5"
    tm = lambda t: np.array([0, 30, 20], np.int64)  # noqa: E731
    s = convert_v2e_h5(src, dst, width=346, height=260, source="v", sensor="v", time_map=tm,
                       simulator_manifest_sha256="0" * 64)
    assert s["nonmonotonic_after_map_sorted"] == 1
    with EventStore(dst) as st:
        np.testing.assert_array_equal(st.events["t"][:], [0, 20, 30])
        np.testing.assert_array_equal(st.events["x"][:], [1, 3, 2])
        np.testing.assert_array_equal(st.events["p"][:], [1, 1, 0])


def test_rejects_bad_layouts(tmp_path):
    bad = tmp_path / "b.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("events", data=np.zeros((3, 4), np.float64))
    with pytest.raises(ValueError, match="uint32"):
        read_v2e_events(bad)
    pol = tmp_path / "p.h5"
    write_fake_v2e(pol, [[0, 0, 0, 2]])
    with pytest.raises(ValueError, match="polarity"):
        read_v2e_events(pol)
    big = tmp_path / "x.h5"
    write_fake_v2e(big, [[0, 400, 0, 1]])
    with pytest.raises(ValueError, match="outside"):
        convert_v2e_h5(big, tmp_path / "o.h5", width=346, height=260, source="v", sensor="v",
                       simulator_manifest_sha256="0" * 64)


def test_chunked_equals_single_chunk_and_clamp_count(fake, tmp_path):
    src, rows = fake
    outs = []
    for i, ch in enumerate((10**9, 333)):
        dst = tmp_path / f"s{i}.h5"
        s = convert_v2e_h5(src, dst, width=346, height=260, source="v", sensor="v",
                           time_map=lambda t: t * 2, time_map_domain_end_us=1_000_000,
                           simulator_manifest_sha256="0" * 64, chunk=ch)
        assert not s["sorted_fallback"]
        assert s["n_events_beyond_clock_map"] == int((rows[:, 0] > 1_000_000).sum())
        with EventStore(dst) as st:
            outs.append({k: st.events[k][:] for k in ("t", "x", "y", "p")})
    for k in outs[0]:
        np.testing.assert_array_equal(outs[0][k], outs[1][k])
    np.testing.assert_array_equal(outs[0]["t"], rows[:, 0] * 2)


def test_fallback_across_chunk_boundary(tmp_path):
    # raw times sorted, but the map reverses order across the chunk boundary only
    rows = np.array([[0, 1, 1, 1], [10, 2, 2, 0], [20, 3, 3, 1], [30, 4, 4, 0]])
    src = tmp_path / "v.h5"
    write_fake_v2e(src, rows)
    tm = lambda t: np.where(t >= 20, t - 15, t)  # noqa: E731  -> 0, 10, 5, 15
    s = convert_v2e_h5(src, tmp_path / "s.h5", width=346, height=260, source="v", sensor="v",
                       time_map=tm, simulator_manifest_sha256="0" * 64, chunk=2)
    assert s["sorted_fallback"] and s["n_events"] == 4
    with EventStore(tmp_path / "s.h5") as st:
        np.testing.assert_array_equal(st.events["t"][:], [0, 5, 10, 15])
        np.testing.assert_array_equal(st.events["x"][:], [1, 3, 2, 4])


def test_rejects_coordinates_that_would_wrap(tmp_path):
    src = tmp_path / "w.h5"
    write_fake_v2e(src, [[0, 65541, 0, 1]])
    with pytest.raises(ValueError, match="outside"):
        convert_v2e_h5(src, tmp_path / "o.h5", width=346, height=260, source="v", sensor="v",
                       simulator_manifest_sha256="0" * 64)
