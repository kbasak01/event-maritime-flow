"""Vectorised tune metrics equal the per-window reference functions (flow_metrics, fwl)."""

import math

import numpy as np
import pytest
import torch
from test_train_data import H, W, synthetic

from evflow.flow.fwl import fwl
from evflow.flow.metrics import flow_metrics
from evflow.train.data import cat_events, collate
from evflow.train.tune_eval import batch_window_metrics


def test_batch_metrics_equal_per_window():
    ds, _ = synthetic(n_frames=8)
    items = [ds[0], ds[2], ds[9]]
    items[1] = {**items[1], **{k: items[1][k][:0] for k in ("x", "y", "t", "p")}}  # no events
    b = collate(items)
    g = torch.Generator().manual_seed(0)
    pred = b["flow"] + torch.randn(b["flow"].shape, generator=g) * 2.0
    valid = torch.rand((3, H, W), generator=g) > 0.2
    crop = torch.zeros((H, W), dtype=torch.bool)
    crop[4:40, 6:60] = True
    x, y, t, _, bi = cat_events(b, "cpu")
    got = batch_window_metrics(pred, b["flow"], valid, crop, (x, y, t, bi), b["duration_us"])
    for j, it in enumerate(items):
        eiw = torch.zeros((H, W), dtype=torch.bool)
        eiw[it["y"], it["x"]] = True
        ref = flow_metrics(pred[j].double(), b["flow"][j].double(), valid_gt=valid[j],
                           events_in_window=eiw, crop=crop)
        for k in ("epe", "outlier_pct", "npe_1", "npe_2", "npe_3", "ae_deg"):
            r, v = ref[k], float(got[k][j])
            assert (math.isnan(r) and math.isnan(v)) or v == pytest.approx(r, abs=1e-9), k
        keep = crop[it["y"], it["x"]]
        if int(keep.sum()) == 0:
            assert math.isnan(float(got["fwl"][j]))
            continue
        r = fwl(it["x"][keep], it["y"][keep], it["t"][keep], pred[j].double(),
                t_range=(0, it["duration_us"]), t_ref=1.0, height=H, width=W,
                interpolation="round")
        assert float(got["fwl"][j]) == pytest.approx(r, abs=1e-9)
        u = pred[j][:, it["y"][keep], it["x"][keep]].double()
        assert float(got["flow_mag"][j]) == pytest.approx(
            float(torch.linalg.vector_norm(u, dim=0).mean()), abs=1e-9)
    assert np.isnan(got["epe"][1].item())
