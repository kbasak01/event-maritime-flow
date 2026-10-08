"""DSEC 16-bit flow PNG decode/encode (checklist A4)."""

import struct
import zlib

import numpy as np
import pytest

from evflow.flow.png import (
    decode_dsec_flow,
    encode_dsec_flow,
    read_dsec_flow_png,
    write_dsec_flow_png,
)


def _png16_rgb(path, rgb: np.ndarray) -> None:
    """Minimal independent PNG writer: 16-bit RGB, channels written in R, G, B order."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].astype(">u2").tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    body = chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)


def test_decode_formula_and_valid():
    rgb = np.zeros((2, 3, 3), np.uint16)
    rgb[..., 0] = 2**15 + 128 * 3      # u = +3
    rgb[..., 1] = 2**15 - 64           # v = -0.5
    rgb[0, 0, 2] = 1
    flow, valid = decode_dsec_flow(rgb)
    assert flow.shape == (2, 2, 3)
    np.testing.assert_allclose(flow[0], 3.0)
    np.testing.assert_allclose(flow[1], -0.5)
    assert valid.tolist() == [[True, False, False], [False, False, False]]


def test_channel_order_from_file(tmp_path):
    rgb = np.zeros((4, 5, 3), np.uint16)
    rgb[..., 0] = 2**15 + 256           # R -> u = +2
    rgb[..., 1] = 2**15 - 128 * 7       # G -> v = -7
    rgb[1:3, 2:4, 2] = 1                # B -> valid
    _png16_rgb(tmp_path / "f.png", rgb)
    flow, valid = read_dsec_flow_png(tmp_path / "f.png")
    np.testing.assert_allclose(flow[0], 2.0)
    np.testing.assert_allclose(flow[1], -7.0)
    assert valid.sum() == 4 and valid[1:3, 2:4].all()


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    flow = np.round(rng.uniform(-200, 200, (2, 6, 7)) * 128) / 128
    valid = rng.random((6, 7)) > 0.3
    write_dsec_flow_png(tmp_path / "r.png", flow, valid)
    f2, v2 = read_dsec_flow_png(tmp_path / "r.png")
    np.testing.assert_array_equal(v2, valid)
    np.testing.assert_allclose(f2[:, valid], flow[:, valid], atol=0)
    rgb = encode_dsec_flow(flow, valid)
    assert rgb.dtype == np.uint16 and np.all(rgb[~valid] == 0)


def test_encode_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode_dsec_flow(np.full((2, 1, 1), 300.0), np.ones((1, 1), bool))
