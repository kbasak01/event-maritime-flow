"""Dataset adapters for evaluation units: window lists, event slicers, GT and masks.

MVSEC windows come from the split manifests (``splits/mvsec_test.json``: ``aps_first`` ..
``aps_last`` per dt; ``splits/mvsec_tune.json``: every window inside its APS range), events from
the canonical stores in ``outputs.events`` (raw pixel grid), GT from ``*_gt_flow_dist.npz`` via
``io.mvsec.gt_flow_for_window`` (EV-FlowNet propagation, D-019). DSEC windows are the
forward-flow GT intervals; events are rectified on read (D-017).

T2 (Phase 6, D-051): ``mvsec_unit(..., events="v2e")`` keeps the windows, GT and region of the
real test unit and reads events from the calibrated v2e(APS) test corpus
(``configs/v2e_corpus_mvsec.yaml`` ``out_root``/test, D-035). Each test window is owned by exactly
one corpus chunk (``owned_window_starts`` in ``corpus_plan.json``, D-032), which also covers its
end; ``ChunkedSlicer`` routes the window there. The sparse mask then comes from the v2e events.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
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


class ChunkedSlicer:
    """Routes a window [t0, t1) to the one chunk store whose owned window starts contain t0.

    ``chunks`` = (first owned start µs, last owned start µs, slicer), inclusive bounds."""

    def __init__(self, chunks: list[tuple[int, int, object]]) -> None:
        self.chunks = sorted(chunks, key=lambda c: c[0])
        for (_, hi, _), (lo, _, _) in zip(self.chunks, self.chunks[1:], strict=False):
            if lo <= hi:
                raise ValueError("chunk ownership overlaps")

    def get_events(self, t_start_us: int, t_end_us: int, **kw) -> dict | None:
        own = [s for lo, hi, s in self.chunks if lo <= t_start_us <= hi]
        if len(own) != 1:
            raise ValueError(f"window start {t_start_us} owned by {len(own)} chunks")
        return own[0].get_events(t_start_us, t_end_us, **kw)


def v2e_test_chunks(paths: dict, sequence: str, aps_us: np.ndarray) -> tuple[ChunkedSlicer, dict]:
    """ChunkedSlicer over the calibrated v2e(APS) test units of ``sequence`` and provenance:
    corpus plan path, unit names, owned ranges and the store SHA-256 recorded by the corpus
    summary (results/v2e_mvsec_corpus.json); callers verify the files against it."""
    cfg = load_yaml("configs/v2e_corpus_mvsec.yaml")
    root = Path(paths["outputs"]["events"]) / cfg["out_root"]
    plan = json.loads((root / "corpus_plan.json").read_text())
    corpus = json.loads((REPO / "results/v2e_mvsec_corpus.json").read_text())
    sha = {u["key"]: u["store_sha256"] for u in corpus["units"]}
    units = [u for u in plan["units"] if u["corpus"] == "test" and u["sequence"] == sequence]
    if not units:
        raise ValueError(f"no v2e test units for {sequence}")
    chunks, stores = [], []
    for u in units:
        k0, k1 = u["owned_window_starts"]
        path = root / "test" / f"{u['name']}.h5"
        chunks.append((int(aps_us[k0]), int(aps_us[k1]), EventSlicer(path)))
        stores.append({"unit": u["key"], "store": f"{cfg['out_root']}/test/{path.name}",
                       "path": str(path), "owned_window_starts": [int(k0), int(k1)],
                       "profile": u["profile"], "seed": u["seed"],
                       "store_sha256": sha[u["key"]]})
    return ChunkedSlicer(chunks), {"corpus_plan": f"{cfg['out_root']}/corpus_plan.json",
                                   "v2e_stores": stores}


def verify_v2e_stores(meta: dict) -> None:
    """T2: every v2e corpus store of a unit must match the SHA-256 recorded by the Phase 3 corpus
    summary; the absolute path is dropped from the provenance (``meta`` is changed in place)."""
    for st in meta.get("v2e_stores", []):
        path = Path(st.pop("path"))
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(1 << 20):
                h.update(block)
        if h.hexdigest() != st["store_sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from results/v2e_mvsec_corpus.json")
        st["store_sha256_verified"] = True


def mvsec_unit(paths: dict, sequence: str, dt: int, *, split: str, region: dict,
               sparse: bool = True, events: str = "real") -> tuple[list[Window], DatasetAccess,
                                                                   dict]:
    if events not in ("real", "v2e"):
        raise ValueError(f"events must be real or v2e, not {events!r}")
    if events == "v2e" and split != "mvsec_test":
        raise ValueError("v2e events are evaluated on mvsec_test only (T2)")
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
    extra: dict = {}
    if events == "real":
        slicer = EventSlicer(store)
    else:
        slicer, extra = v2e_test_chunks(paths, sequence, aps)
        owned = sorted(r for st in extra["v2e_stores"] for r in [st["owned_window_starts"]])
        if owned[0][0] > first or owned[-1][1] < last:
            raise ValueError(f"v2e chunks {owned} do not own windows {first}..{last}")
    data = DatasetAccess(
        slicer=slicer,
        gt=lambda w: mvsec_io.gt_flow_for_window(gt, w.t_start_us, w.t_end_us),
        height=mvsec_io.HEIGHT, width=mvsec_io.WIDTH, region=region, sparse=sparse)
    meta = {"split": split, "split_manifest_sha256": doc.get("manifest_sha256"),
            "aps_first": first, "aps_last": last, "events": events,
            **({"events_source": "v2e(APS) mvsec_cal test corpus (D-035)"} if events == "v2e"
               else {}),
            **({"store": store.name} if events == "real" else extra)}
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
