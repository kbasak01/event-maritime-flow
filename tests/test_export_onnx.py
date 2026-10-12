"""ONNX export of M1 (Phase 8 step 1): opset 17, finest flow only, static shapes, fp16 IO for the
fp16 graph, ORT CPU parity with torch on a small net, and the parity statistics (step 2)."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from evflow.export.onnx import (
    INPUT_NAME,
    OPSET,
    OUTPUT_NAME,
    FinestFlow,
    export_m1,
    padded_hw,
    parity_stats,
)
from evflow.models.evflownet import EVFlowNet


def _net(cin=5):
    torch.manual_seed(0)
    net = EVFlowNet(in_channels=cin, base_channels=8)
    # non-trivial BatchNorm statistics, so the exported graph is not the identity-BN case
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    return net.eval()


def test_wrapper_returns_finest_flow():
    net = _net()
    x = torch.randn(1, 5, 64, 96)
    with torch.no_grad():
        assert torch.equal(FinestFlow(net)(x), net(x)[-1])


def test_padded_hw():
    assert padded_hw(260, 346) == (272, 352)
    assert padded_hw(256, 256) == (256, 256)
    assert padded_hw(480, 640) == (480, 640)


def test_export_fp32_matches_torch_on_ort_cpu(tmp_path):
    net = _net()
    path = tmp_path / "m1.onnx"
    info = export_m1(net, path, in_channels=5, height=64, width=96, fp16=False,
                     meta={"run_id": "unit"})
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    assert {o.domain: o.version for o in model.opset_import}[""] == OPSET == 17
    (inp,), (out,) = model.graph.input, model.graph.output
    assert inp.name == INPUT_NAME and out.name == OUTPUT_NAME
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, 5, 64, 96]
    assert [d.dim_value for d in out.type.tensor_type.shape.dim] == [1, 2, 64, 96]
    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    side = json.loads(path.with_suffix(".json").read_text())
    assert side == info
    assert side["dynamic_axes"] is None and side["opset"] == 17 and side["run_id"] == "unit"
    assert len(side["onnx_sha256"]) == 64

    x = torch.randn(1, 5, 64, 96)
    with torch.no_grad():
        ref = net(x)[-1].numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {INPUT_NAME: x.numpy()})[0]
    assert np.abs(got - ref).max() < 1e-4


def test_export_fp16_has_half_io(tmp_path):
    net = _net()
    path = tmp_path / "m1_fp16.onnx"
    export_m1(net, path, in_channels=5, height=32, width=32, fp16=True)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    for v in (*model.graph.input, *model.graph.output):
        assert v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    # the caller's model is not converted in place
    assert next(net.parameters()).dtype == torch.float32


def test_export_rejects_unpadded_shape(tmp_path):
    with pytest.raises(ValueError):
        export_m1(_net(), tmp_path / "x.onnx", in_channels=5, height=260, width=346, fp16=False)


def test_parity_stats():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((3, 2, 4, 5))
    test = ref.copy()
    test[1, 0, 2, 3] += 0.3
    test[1, 1, 2, 3] -= 0.4            # |delta| = 0.5 at one pixel
    test[2, 0, 0, 0] += 0.1            # outside the mask
    mask = np.ones((3, 4, 5), bool)
    mask[2, 0, 0] = False
    s = parity_stats(ref, test, mask=mask)
    assert s["max_px"] == pytest.approx(0.5)
    assert s["max_component_px"] == pytest.approx(0.4)
    assert s["max_px_masked"] == pytest.approx(0.5)
    assert s["mean_px"] == pytest.approx(0.6 / 60)
    assert s["n_px"] == 60 and s["n_px_masked"] == 59
    s2 = parity_stats(ref, test, mask=np.zeros((3, 4, 5), bool) | (np.arange(5) == 0))
    assert s2["max_px_masked"] == pytest.approx(0.1)
    assert s2["max_px"] == pytest.approx(0.5)


def test_parity_stats_non_finite_is_infinite():
    ref = np.zeros((1, 2, 2, 2))
    test = ref.copy()
    test[0, 0, 0, 0] = np.nan
    s = parity_stats(ref, test, mask=np.ones((1, 2, 2), bool))
    assert s["max_px"] == np.inf and s["n_non_finite"] == 1
