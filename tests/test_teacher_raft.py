"""RAFT teacher: forward-backward consistency (numpy reference vs torch) and the wrapper's
shapes, on CPU with an untrained raft_small (no weight download). Phase 7 step 2, D-055."""

import numpy as np
import pytest
import torch

from evflow.teacher import raft

A1, A2 = 0.01, 0.5


def _const(h, w, u, v):
    f = np.zeros((2, h, w))
    f[0], f[1] = u, v
    return f


def test_translation_is_consistent_inside_the_image():
    h, w = 20, 30
    m = raft.fb_consistency_mask_ref(_const(h, w, 2.0, 1.0), _const(h, w, -2.0, -1.0), A1, A2)
    exp = np.zeros((h, w), bool)
    exp[: h - 1, : w - 2] = True          # x + 2 <= w - 1 and y + 1 <= h - 1
    np.testing.assert_array_equal(m, exp)


def test_inconsistent_backward_flow_is_rejected():
    h, w = 20, 30
    f = _const(h, w, 2.0, 0.0)
    b = _const(h, w, -2.0, 0.0)
    b[:, 5:10, 10:20] = 0.0               # an occluder: backward flow disagrees there
    m = raft.fb_consistency_mask_ref(f, b, A1, A2)
    # pixels landing (x + 2) inside the occluded box fail: |f + b|^2 = 4 > 0.01 * 4 + 0.5
    assert not m[5:10, 8:18].any()
    assert m[12:, :20].all()


def test_threshold_boundary_squared_form():
    # |f + b|^2 vs alpha1 (|f|^2 + |b|^2) + alpha2 with f = (10, 0), b = (-10 + e, 0)
    h, w = 4, 40
    f = _const(h, w, 10.0, 0.0)
    for e, ok in ((0.70, True), (1.60, False)):
        b = _const(h, w, -10.0 + e, 0.0)
        # rhs = 0.01 (100 + (10 - e)^2) + 0.5 : e = 0.7 -> 2.36 > 0.49 ; e = 1.6 -> 2.21 < 2.56
        m = raft.fb_consistency_mask_ref(f, b, A1, A2)
        assert bool(m[0, 0]) is ok
    # the literal (unsquared) form of §1.6 accepts e = 1.6 (1.6 < 2.21)
    b = _const(h, w, -8.4, 0.0)
    assert raft.fb_consistency_mask_literal_ref(f, b, A1, A2)[0, 0]


def test_torch_equals_reference():
    rng = np.random.default_rng(0)
    f = np.empty((3, 2, 24, 32))
    f[:, 0], f[:, 1] = 2.3, -1.4                       # a smooth (constant) motion + noise
    f += rng.normal(0, 0.3, f.shape)
    b = -f.mean(axis=(2, 3), keepdims=True) + rng.normal(0, 0.5, f.shape)
    got = raft.fb_consistency_mask(torch.from_numpy(f), torch.from_numpy(b), A1, A2).numpy()
    ref = np.stack([raft.fb_consistency_mask_ref(f[i], b[i], A1, A2) for i in range(3)])
    assert 0.2 < ref.mean() < 0.9
    np.testing.assert_array_equal(got, ref)


def test_pad_to_multiple():
    x = torch.arange(2 * 3 * 10 * 13, dtype=torch.float32).reshape(2, 3, 10, 13)
    y, (h, w) = raft.pad_to_multiple(x, 8)
    assert y.shape == (2, 3, 16, 16) and (h, w) == (10, 13)
    assert torch.equal(y[..., :10, :13], x) and torch.equal(y[..., 15, 15], x[..., 9, 12])


def test_wrapper_shapes_untrained_raft_small():
    from torchvision.models.optical_flow import raft_small

    torch.manual_seed(0)
    t = raft.RaftTeacher(device="cpu", model=raft_small(weights=None), iters=2)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (128, 136, 3), dtype=np.uint8) for _ in range(4)]
    out = t.pairs(frames, A1, A2, batch=2)
    assert out["flow"].shape == (3, 2, 128, 136) and out["flow"].dtype == np.float32
    assert out["valid"].shape == (3, 128, 136) and out["valid"].dtype == bool
    assert out["valid_literal"].shape == (3, 128, 136)


@pytest.mark.gpu
def test_raft_large_recovers_a_shift():
    t = raft.RaftTeacher(device="cuda")
    rng = np.random.default_rng(1)
    big = (rng.random((400, 700, 3)) * 255).astype(np.uint8)
    import cv2
    big = cv2.GaussianBlur(big, (0, 0), 2.0)
    a, b = big[20:380, 20:660], big[17:377, 15:655]      # content moves by (+5, +3)
    out = t.pairs([a, b], A1, A2)
    med = np.median(out["flow"][0][:, 40:-40, 40:-40].reshape(2, -1), axis=1)
    np.testing.assert_allclose(med, [5.0, 3.0], atol=0.2)
