"""Dataset adapters for evaluation units: window lists, event slicers, GT and masks.

MVSEC windows come from the split manifests (``splits/mvsec_test.json``: ``aps_first`` ..
``aps_last`` per dt; ``splits/mvsec_tune.json``: every window inside its APS range), events from
the canonical stores in ``outputs.events`` (raw pixel grid), GT from ``*_gt_flow_dist.npz`` via
``io.mvsec.gt_flow_for_window`` (EV-FlowNet propagation, D-019). DSEC windows are the
forward-flow GT intervals; events are rectified on read (D-017).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from evflow.eval.run import DatasetAccess
from evflow.eval.windows import Window, dsec_windows, mvsec_windows
from evflow.io import dsec as dsec_io
from evflow.io import mvsec as mvsec_io
from evflow.io.store import EventSlicer
from evflow.sim.splits import load_split

REPO = Path(__file__).resolve().parents[3]


def load_yaml(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())


def mvsec_scene(sequence: str) -> str:
    m = re.fullmatch(r"([a-z_]+?)(\d+)", sequence)
    if not m:
        raise ValueError(f"not an MVSEC sequence name: {sequence}")
    return m.group(1)


def mvsec_unit(paths: dict, sequence: str, dt: int, *, split: str, region: dict,
               sparse: bool = True) -> tuple[list[Window], DatasetAccess, dict]:
    doc = load_split(split)
    scene = mvsec_scene(sequence)
    src = Path(paths["mvsec"]["root"]) / scene / f"{sequence}_data.hdf5"
    aps = mvsec_io.read_aps_timestamps_us(src)
    if "sequences" in doc:   # evaluation split: explicit window ranges per dt
        s = next((s for s in doc["sequences"] if s["sequence"] == sequence), None)
        if s is None:
            raise ValueError(f"{sequence} not in {split}")
        r = s["windows"][f"dt{dt}"]
        first, last, expected = int(r["aps_first"]), int(r["aps_last"]), int(r["n_windows"])
    else:                    # train/tune split: every window inside each APS range
        if doc["sequence"] != sequence or len(doc["aps"]) != 1:
            raise ValueError(f"{split}: expected one APS range of {sequence}")
        a = doc["aps"][0]
        first, last = int(a["aps_first"]), int(a["aps_last"]) - dt
        expected = int(a[f"n_windows_dt{dt}"])
    windows = mvsec_windows(aps, aps_first=first, aps_last=last, dt=dt)
    if len(windows) != expected:
        raise ValueError(f"{split}/{sequence} dt{dt}: {len(windows)} windows != {expected}")
    gt = mvsec_io.GTFlow(Path(paths["mvsec"]["gt_flow"]) / scene / f"{sequence}_gt_flow_dist.npz")
    store = Path(paths["outputs"]["events"]) / f"mvsec_{sequence}_left.h5"
    slicer = EventSlicer(store)
    data = DatasetAccess(
        slicer=slicer,
        gt=lambda w: mvsec_io.gt_flow_for_window(gt, w.t_start_us, w.t_end_us),
        height=mvsec_io.HEIGHT, width=mvsec_io.WIDTH, region=region, sparse=sparse)
    meta = {"split": split, "split_manifest_sha256": doc.get("manifest_sha256"),
            "aps_first": first, "aps_last": last, "store": store.name}
    return windows, data, meta


def dsec_unit(paths: dict, sequence: str, *, delta_t_us: int) -> tuple[list[Window],
                                                                        DatasetAccess, dict]:
    doc = load_split("dsec_test")
    entry = next((s for s in doc["sequences"] if s["sequence"] == sequence), None)
    if entry is None:
        raise ValueError(f"{sequence} not in dsec_test")
    gt = dsec_io.FlowGT(Path(paths["dsec"]["root"]) / sequence)
    if len(gt) != int(entry["flow_png_count"]):
        raise ValueError(f"{sequence}: {len(gt)} flow maps != manifest {entry['flow_png_count']}")
    windows = dsec_windows(gt.timestamps_us, delta_t_us=delta_t_us)
    store = Path(paths["outputs"]["events"]) / f"dsec_{sequence}_left.h5"
    slicer = EventSlicer(store)

    def gt_fn(w: Window):
        flow, valid, _ = gt[w.index]
        return flow, valid

    data = DatasetAccess(slicer=slicer, gt=gt_fn, height=dsec_io.HEIGHT, width=dsec_io.WIDTH,
                         region=None, sparse=False)
    meta = {"split": "dsec_test", "split_manifest_sha256": doc.get("manifest_sha256"),
            "store": store.name}
    return windows, data, meta
