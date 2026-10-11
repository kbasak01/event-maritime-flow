"""Training and tune data for the MVSEC arms (IMPLEMENTATION_PLAN Phase 5, §1.4; D-026, D-032).

Every arm trains on the same windows: the 4 x 60 s D-026 clips of ``mvsec_train``
(``sim.corpus.train_clips``, cross-checked against the corpus plan), first ``warmup`` windows of
each clip dropped (D-032c). Window k with dt = d spans [aps[k], aps[k + d]] and must end inside
its clip. Tune windows: the whole ``mvsec_tune`` slice, first ``warmup`` windows dropped.

Event sources (``events`` in configs/train_<arm>.yaml):
    real       canonical real store of outdoor_day2
    train_cal  v2e(APS) with the calibrated profile (one store per clip)
    rand       v2e(APS) with randomised CT: K stores per clip; each sample draws one of them
    v2e        (tune only) the calibrated v2e(tune) store

GT comes from the per-window cache written by ``scripts/cache_mvsec_gt.py`` (EV-FlowNet
propagation, ``io.mvsec.gt_flow_for_window``). A split guard refuses anything outside
``mvsec_train`` / ``mvsec_tune``: training and selection never read a test split (CLAUDE.md
rule 4).

Augmentation (training only): random crop, horizontal flip (x -> w - 1 - x, u -> -u,
docs/CONVENTIONS.md), polarity flip off by default. Rows at or below ``hood_row`` (full-frame
coordinates, outdoor car hood, as ``center_256_no_hood``) are invalid for supervision and their
events are excluded from the contrast-maximisation loss.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch

from evflow.events.representations import count_image, time_surface, voxel_grid
from evflow.sim.splits import LeakageError, check_aps_in_split

REPO = Path(__file__).resolve().parents[3]
SEQUENCE = "outdoor_day2"
HEIGHT, WIDTH = 260, 346
ALLOWED_SPLITS = ("mvsec_train", "mvsec_tune")
REPRESENTATIONS = ("voxel5", "voxel15", "timesurface", "count")


# ---------------------------------------------------------------------------------------------
# windows

@dataclass(frozen=True)
class Clip:
    name: str
    split: str
    aps_first: int     # first APS frame of the clip (inclusive)
    aps_last: int      # last APS frame of the clip (inclusive); windows end at or before it


def guard_split(split: str, clip: Clip) -> None:
    from evflow.sim.splits import load_split
    if split not in ALLOWED_SPLITS:
        raise LeakageError(f"training/selection may read {ALLOWED_SPLITS} only, not {split!r}")
    check_aps_in_split(load_split(split), SEQUENCE, clip.aps_first, clip.aps_last)


def clip_window_starts(clip: Clip, dt: int, warmup: int) -> np.ndarray:
    """APS start frames k of the dt-windows of a clip: k in [first + warmup, last - dt]."""
    k0, k1 = clip.aps_first + warmup, clip.aps_last - dt
    if k1 < k0:
        raise ValueError(f"{clip.name}: no dt={dt} window after {warmup} warm-up windows")
    return np.arange(k0, k1 + 1, dtype=np.int64)


def train_clips_from_split(split_doc: dict, aps_us: np.ndarray, corpus_cfg: dict) -> list[Clip]:
    """The D-026 clips, computed from the committed config + split manifest."""
    from evflow.sim.corpus import train_clips
    t = corpus_cfg["train"]
    ranges = train_clips(split_doc, aps_us, int(t["n_clips"]), float(t["clip_s"]),
                         int(t["range_index"]))
    return [Clip(f"train_clip{i}", "mvsec_train", a, b) for i, (a, b) in enumerate(ranges)]


def tune_clip(split_doc: dict) -> Clip:
    a = split_doc["aps"][0]
    return Clip(f"tune_{SEQUENCE}", "mvsec_tune", int(a["aps_first"]), int(a["aps_last"]))


def check_against_corpus_plan(clips: list[Clip], plan: dict, corpus: str) -> list[dict]:
    """The v2e units of ``corpus`` must cover exactly these clips; returns those units."""
    units = [u for u in plan["units"] if u["corpus"] == corpus]
    by_name = {u["name"]: u for u in units}
    out = []
    for c in clips:
        u = by_name.get(c.name)
        if u is None or [int(v) for v in u["aps"]] != [c.aps_first, c.aps_last]:
            raise ValueError(f"corpus {corpus}: no unit with APS [{c.aps_first}, {c.aps_last}] "
                             f"named {c.name}")
        out.append(u)
    return out


# ---------------------------------------------------------------------------------------------
# events in memory

@dataclass
class EventArrays:
    """Events of one clip in RAM (absolute µs), sliced by binary search."""

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    source: str = ""
    end_clamped_us: int = 0

    def window(self, t0: int, t1: int) -> dict[str, np.ndarray]:
        if t0 < int(self.t[0]) - 10_000 or t1 > int(self.t[-1]) + 10_000:
            raise ValueError(f"window [{t0}, {t1}) outside the events of {self.source}")
        a, b = np.searchsorted(self.t, [t0, t1], side="left")
        return {"x": self.x[a:b], "y": self.y[a:b], "t": self.t[a:b], "p": self.p[a:b]}

    @classmethod
    def from_store(cls, path: str | os.PathLike, t0: int, t1: int,
                   max_end_shortfall_us: int = 1000) -> EventArrays:
        """Events in [t0, t1). A v2e store whose last event lies just before the clip's last
        APS frame has an ``ms_to_idx`` ending short of t1, and the slicer then refuses the
        range; if the shortfall is at most ``max_end_shortfall_us`` the range is clamped to the
        store's covered end (no events exist past it) and ``end_clamped_us`` records it (D-048)."""
        from evflow.io.store import EventSlicer
        clamp = 0
        with EventSlicer(path) as s:
            ev = s.get_events(int(t0), int(t1), rectify=False)
            if ev is None:
                covered = s.store.t_offset + (s.store.ms_to_idx.size - 1) * 1000
                if 0 < t1 - covered <= max_end_shortfall_us and s.t_final_us < covered:
                    clamp = int(t1 - covered)
                    ev = s.get_events(int(t0), int(covered), rectify=False)
        if ev is None:
            raise ValueError(f"{path}: [{t0}, {t1}) not covered")
        out = cls(ev["x"], ev["y"], ev["t"], ev["p"].astype(np.int8), source=str(path))
        out.end_clamped_us = clamp
        return out


# ---------------------------------------------------------------------------------------------
# GT cache

def gt_cache_path(root: str | os.PathLike, split: str, dt: int) -> Path:
    return Path(root) / f"{split}_dt{dt}.h5"


class GTCache:
    """Per-window GT (``flow`` (N, 2, H, W) float32, ``valid`` (N, H, W) bool) keyed by the
    window's APS start frame. The HDF5 file is opened lazily per process (fork-safe)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.aps_start = f["aps_start"][:]
            self.dt = int(f.attrs["dt"])
            self.split = str(f.attrs["split"])
        self._index = {int(k): i for i, k in enumerate(self.aps_start)}
        self._f: h5py.File | None = None
        self._pid = -1

    def __contains__(self, k: int) -> bool:
        return int(k) in self._index

    def get(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._f is None or self._pid != os.getpid():
            self._f = h5py.File(self.path, "r")
            self._pid = os.getpid()
        i = self._index[int(k)]
        return self._f["flow"][i], self._f["valid"][i]

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f"], d["_pid"] = None, -1
        return d


class GTArray:
    """In-memory GT with the ``GTCache`` interface (tests, smoke fixtures)."""

    def __init__(self, items: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self.items = items

    def __contains__(self, k: int) -> bool:
        return int(k) in self.items

    def get(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.items[int(k)]


# ---------------------------------------------------------------------------------------------
# datasets

@dataclass
class ClipData:
    clip: Clip
    sources: list[EventArrays]        # one (real, train_cal, tune) or K (rand)


@dataclass
class WindowRef:
    clip: int
    dt: int
    k: int
    t0: int
    t1: int


def build_windows(clips: list[ClipData], aps_us: np.ndarray, dts, warmup: int,
                  gts: dict[int, object]) -> dict[int, list[WindowRef]]:
    out = {}
    for dt in dts:
        refs = []
        for ci, cd in enumerate(clips):
            for k in clip_window_starts(cd.clip, dt, warmup):
                if int(k) not in gts[dt]:
                    raise KeyError(f"GT cache dt={dt} has no window k={int(k)} ({cd.clip.name})")
                refs.append(WindowRef(ci, dt, int(k), int(aps_us[k]), int(aps_us[k + dt])))
        out[dt] = refs
    return out


def thin(windows: dict[int, list[WindowRef]], n: int | None) -> dict[int, list[WindowRef]]:
    """At most ``n`` evenly spaced windows per dt (smoke runs); ``None`` keeps all."""
    if n is None:
        return windows
    out = {}
    for dt, v in windows.items():
        idx = np.unique(np.linspace(0, len(v) - 1, min(n, len(v))).round().astype(int))
        out[dt] = [v[i] for i in idx]
    return out


@dataclass
class Augment:
    crop: tuple[int, int] | None = (256, 256)
    hflip: bool = True
    polarity_flip: bool = False
    fixed_crop: tuple[int, int] | None = None   # (r0, c0): deterministic crop (smoke)


class WindowDataset(torch.utils.data.Dataset):
    """Items: events of one window (sample frame, int64 µs relative to window start), GT flow
    (2, h, w), supervision mask (h, w), the CM event mask, dt and provenance. Index space =
    the windows of every dt, concatenated in ``dts`` order (see ``mixed_dt_weights``)."""

    def __init__(self, clips: list[ClipData], windows: dict[int, list[WindowRef]],
                 gts: dict[int, object], *, augment: Augment | None, hood_row: int | None,
                 height: int = HEIGHT, width: int = WIDTH) -> None:
        self.clips, self.gts, self.aug = clips, gts, augment
        self.hood_row, self.h, self.w = hood_row, height, width
        self.refs = [r for dt in windows for r in windows[dt]]
        self.dt_sizes = {dt: len(v) for dt, v in windows.items()}

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> dict:
        r = self.refs[i]
        cd = self.clips[r.clip]
        src = 0
        if len(cd.sources) > 1:
            src = int(torch.randint(len(cd.sources), (1,)))
        ev = cd.sources[src].window(r.t0, r.t1)
        flow, valid = self.gts[r.dt].get(r.k)
        x = ev["x"].astype(np.int64)
        y = ev["y"].astype(np.int64)
        t = ev["t"].astype(np.int64) - r.t0
        p = ev["p"].astype(np.int64)
        flow = np.array(flow, np.float32)
        valid = np.array(valid, bool)
        rows = np.arange(self.h)
        sup = valid & ((rows < self.hood_row)[:, None] if self.hood_row is not None else True)
        cm_keep = y < self.hood_row if self.hood_row is not None else np.ones(y.shape, bool)
        h, w = self.h, self.w
        a = self.aug
        if a is not None and (a.crop is not None or a.fixed_crop is not None):
            ch, cw = a.crop if a.crop is not None else (h, w)
            if a.fixed_crop is not None:
                r0, c0 = a.fixed_crop
            else:
                r0 = int(torch.randint(h - ch + 1, (1,)))
                c0 = int(torch.randint(w - cw + 1, (1,)))
            keep = (x >= c0) & (x < c0 + cw) & (y >= r0) & (y < r0 + ch)
            x, y, t, p, cm_keep = x[keep] - c0, y[keep] - r0, t[keep], p[keep], cm_keep[keep]
            flow = flow[:, r0:r0 + ch, c0:c0 + cw]
            sup = sup[r0:r0 + ch, c0:c0 + cw]
            h, w = ch, cw
        if a is not None and a.hflip and a.fixed_crop is None and bool(torch.rand(()) < 0.5):
            x = w - 1 - x
            flow = flow[:, :, ::-1].copy()
            flow[0] = -flow[0]
            sup = sup[:, ::-1].copy()
        if a is not None and a.polarity_flip and bool(torch.rand(()) < 0.5):
            p = 1 - p
        return {"x": torch.from_numpy(np.ascontiguousarray(x)),
                "y": torch.from_numpy(np.ascontiguousarray(y)),
                "t": torch.from_numpy(t), "p": torch.from_numpy(p),
                "cm_keep": torch.from_numpy(cm_keep),
                "duration_us": r.t1 - r.t0,
                "flow": torch.from_numpy(np.ascontiguousarray(flow)),
                "mask": torch.from_numpy(np.ascontiguousarray(sup)),
                "dt": r.dt, "k": r.k, "clip": r.clip, "source": src}


def collate(items: list[dict]) -> dict:
    """Packs the variable-length events of a batch into one tensor per field (``ev_x``, ``ev_y``,
    ``ev_t``, ``ev_p``, ``ev_cm_keep``) plus ``ev_b`` (sample index) and ``n_events``: a handful
    of tensors per batch keeps DataLoader worker IPC cheap."""
    out = {f"ev_{k}": torch.cat([it[k] for it in items]) for k in ("x", "y", "t", "p", "cm_keep")}
    n = torch.tensor([it["x"].numel() for it in items], dtype=torch.long)
    out["n_events"] = n
    out["ev_b"] = torch.repeat_interleave(torch.arange(len(items)), n)
    out["flow"] = torch.stack([it["flow"] for it in items])
    out["mask"] = torch.stack([it["mask"] for it in items])
    for k in ("duration_us", "dt", "k", "clip", "source"):
        out[k] = torch.tensor([it[k] for it in items], dtype=torch.long)
    return out


def mixed_dt_weights(dt_sizes: dict[int, int], fractions: dict[int, float]) -> torch.Tensor:
    """Per-item sampling weights so that dt d is drawn with probability ``fractions[d]``,
    uniformly over its windows."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or set(fractions) != set(dt_sizes):
        raise ValueError(f"fractions {fractions} must cover {sorted(dt_sizes)} and sum to 1")
    return torch.cat([torch.full((n,), fractions[dt] / n, dtype=torch.float64)
                      for dt, n in dt_sizes.items()])


# ---------------------------------------------------------------------------------------------
# network input

def in_channels(rep: str) -> int:
    if rep.startswith("voxel"):
        return int(rep.removeprefix("voxel"))
    if rep in ("timesurface", "count"):
        return 2
    raise ValueError(f"unknown representation {rep!r}, expected one of {REPRESENTATIONS}")


def make_input(rep: str, x, y, t, p, *, duration_us: int, height: int, width: int,
               tau_fraction: float = 1.0 / 3.0) -> torch.Tensor:
    """Representation of one window; ``t`` is relative to the window start (t_range =
    (0, duration_us), explicit as docs/CONVENTIONS.md requires). §1.5 definitions: voxel = signed,
    bilinear in time, normalised over non-zero cells; time surface = per-polarity exponential
    decay, tau = window/3; count = per-polarity counts."""
    tr = (0, int(duration_us))
    if rep.startswith("voxel"):
        return voxel_grid(x, y, t, p, bins=in_channels(rep), height=height, width=width,
                          t_range=tr, normalize=True)
    if rep == "timesurface":
        return time_surface(x, y, t, p, height=height, width=width, t_range=tr,
                            tau_us=tau_fraction * duration_us)
    if rep == "count":
        return count_image(x, y, p, height=height, width=width)
    raise ValueError(f"unknown representation {rep!r}")


def cat_events(batch: dict, device):
    """(x, y, t, p, b) of a collated batch as int64 tensors on ``device``."""
    return tuple(batch[f"ev_{k}"].to(device, non_blocking=True).long()
                 for k in ("x", "y", "t", "p", "b"))


def batch_inputs(batch: dict, rep: str, height: int, width: int, device) -> torch.Tensor:
    """(B, C, H, W) representations of a batch in one scatter per tap; equal to stacking
    ``make_input`` per sample (tests/test_train_data.py). Integer pixel coordinates only (MVSEC
    raw grid)."""
    b_n = int(batch["n_events"].numel())
    x, y, t, p, b = cat_events(batch, device)
    dur = batch["duration_us"].to(device).long()
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if rep.startswith("voxel"):
        bins = in_channels(rep)
        tn = (bins - 1) * t.float() / dur[b].float()       # as representations._normalize_time
        vox = torch.zeros(b_n * bins * height * width, dtype=torch.float32, device=device)
        t0 = torch.floor(tn)
        pol = 2.0 * p.float() - 1.0
        for d in (0, 1):
            ti = t0 + d
            wt = torch.clamp(1.0 - (ti - tn).abs(), min=0.0)
            good = ok & (ti >= 0) & (ti < bins)
            idx = ((b * bins + ti.long()) * height + y) * width + x
            vox.index_put_((idx[good],), (pol * wt)[good], accumulate=True)
        vox = vox.view(b_n, bins * height * width)
        nz = vox != 0
        n = nz.sum(1)
        mean = (vox * nz).sum(1) / n.clamp(min=1)
        var = (((vox - mean[:, None]) * nz) ** 2).sum(1) / (n - 1).clamp(min=1)
        std = torch.where(n > 1, var.sqrt(), torch.zeros_like(var))
        scale = torch.where(std > 0, std, torch.ones_like(std))
        vox = torch.where(nz, (vox - mean[:, None]) / scale[:, None], vox)
        return vox.view(b_n, bins, height, width)
    if rep == "count":
        img = torch.zeros(b_n * 2 * height * width, dtype=torch.float32, device=device)
        idx = ((b * 2 + p) * height + y) * width + x
        img.index_put_((idx[ok],), torch.ones_like(idx[ok], dtype=torch.float32),
                       accumulate=True)
        return img.view(b_n, 2, height, width)
    if rep == "timesurface":
        flat = ((b * 2 + p) * height + y) * width + x
        neg_age = -(dur[b] - t).double()
        last = torch.full((b_n * 2 * height * width,), -float("inf"), dtype=torch.float64,
                          device=device)
        last.scatter_reduce_(0, flat[ok], neg_age[ok], reduce="amax")
        out = torch.zeros_like(last)
        has = torch.isfinite(last)
        tau = (dur.double() / 3.0).repeat_interleave(2 * height * width)
        out[has] = torch.exp(last[has] / tau[has])
        return out.view(b_n, 2, height, width).float()
    raise ValueError(f"unknown representation {rep!r}")


# ---------------------------------------------------------------------------------------------
# MVSEC assembly (needs configs/paths.local.yaml and the data)

@dataclass
class MvsecData:
    train: WindowDataset | None
    tune: dict[str, WindowDataset]          # stream name -> full-frame tune dataset
    provenance: dict = field(default_factory=dict)


def _yaml(p: Path) -> dict:
    import yaml
    return yaml.safe_load(p.read_text())


def _sha(path: Path) -> str:
    from evflow.models.eraft_runner import sha256_file
    return sha256_file(path)


def load_mvsec(*, train_source: str | None, tune_streams: list[str], dts, warmup: int,
               hood_row: int, augment: Augment | None, gt_root: Path | None = None,
               max_train_windows: int | None = None,
               max_tune_windows: int | None = None) -> MvsecData:
    """Assemble the train dataset (``train_source`` in {real, train_cal, rand}, or None) and the
    full-frame tune datasets (streams in {real, v2e})."""
    from evflow.io import mvsec as mvsec_io
    from evflow.sim.splits import load_split
    paths = _yaml(REPO / "configs" / "paths.local.yaml")
    corpus_cfg = _yaml(REPO / "configs" / "v2e_corpus_mvsec.yaml")
    events_root = Path(paths["outputs"]["events"])
    v2e_root = events_root / corpus_cfg["out_root"]
    gt_root = Path(gt_root) if gt_root else events_root / "gt_cache"
    real_store = events_root / f"mvsec_{SEQUENCE}_left.h5"
    aps = mvsec_io.read_aps_timestamps_us(
        Path(paths["mvsec"]["root"]) / "outdoor_day" / f"{SEQUENCE}_data.hdf5")
    plan_path = v2e_root / "corpus_plan.json"
    plan = json.loads(plan_path.read_text())
    prov: dict = {"aps_frames": int(aps.size), "corpus_plan": str(plan_path.relative_to(
        events_root)), "corpus_plan_sha256": _sha(plan_path), "gt_cache": {}, "stores": {}}

    def gt_for(split: str) -> dict[int, GTCache]:
        """GT caches of ``split``; each must match the current split manifest and warm-up."""
        split_sha = load_split(split)["manifest_sha256"]
        out = {}
        for dt in dts:
            p = gt_cache_path(gt_root, split, dt)
            out[dt] = GTCache(p)
            if out[dt].split != split or out[dt].dt != dt:
                raise LeakageError(f"{p}: holds {out[dt].split} dt{out[dt].dt}")
            man = json.loads(p.with_suffix(".json").read_text())
            built = (man.get("split_manifest_sha256"), int(man.get("warmup", -1)))
            if built != (split_sha, warmup):
                raise LeakageError(f"{p}: built for (split manifest, warm-up) {built}, "
                                   f"current ({split_sha}, {warmup})")
            prov["gt_cache"][p.name] = man.get("sha256")
        return out

    def store_entry(path: Path) -> None:
        man = path.with_suffix(".json")
        doc = json.loads(man.read_text()) if man.exists() else {}
        prov["stores"][str(path.relative_to(events_root))] = {
            "manifest_sha256": _sha(man) if man.exists() else None,
            "n_events": doc.get("n_events"), "seed": doc.get("seed"),
            "overrides": doc.get("overrides")}

    train = None
    if train_source is not None:
        split = load_split("mvsec_train")
        clips = train_clips_from_split(split, aps, corpus_cfg)
        for c in clips:
            guard_split("mvsec_train", c)
        prov["train_split_manifest_sha256"] = split["manifest_sha256"]
        prov["train_clips"] = [[c.aps_first, c.aps_last] for c in clips]
        check_against_corpus_plan(clips, plan, "train_cal")
        data = []
        n_rand = int(_yaml(REPO / "configs" / "v2e_profiles.yaml")["profiles"]["mvsec_rand"]
                     ["randomise"]["corpora"])
        for c in clips:
            t0, t1 = int(aps[c.aps_first + warmup]), int(aps[c.aps_last])
            if train_source == "real":
                stores = [real_store]
            elif train_source == "train_cal":
                stores = [v2e_root / "train_cal" / f"{c.name}.h5"]
            elif train_source == "rand":
                stores = [v2e_root / f"rand{j}" / f"{c.name}.h5" for j in range(n_rand)]
                units = [u for j in range(n_rand) for u in
                         check_against_corpus_plan([c], plan, f"rand{j}")]
                prov.setdefault("rand_draws", {})[c.name] = [
                    {"corpus": u["corpus"], "seed": u["seed"], "overrides": u["overrides"]}
                    for u in units]
            else:
                raise ValueError(f"unknown train source {train_source!r}")
            for s in stores:
                store_entry(s)
            arrays = [EventArrays.from_store(s, t0, t1) for s in stores]
            for s_, a_ in zip(stores, arrays, strict=True):
                if a_.end_clamped_us:
                    prov.setdefault("end_clamped_us", {})[
                        str(s_.relative_to(events_root))] = a_.end_clamped_us
            data.append(ClipData(c, arrays))
        gts = gt_for("mvsec_train")
        windows = build_windows(data, aps, dts, warmup, gts)
        windows = thin(windows, max_train_windows)
        train = WindowDataset(data, windows, gts, augment=augment, hood_row=hood_row)

    tune = {}
    if tune_streams:
        split = load_split("mvsec_tune")
        c = tune_clip(split)
        guard_split("mvsec_tune", c)
        prov["tune_split_manifest_sha256"] = split["manifest_sha256"]
        prov["tune_clip"] = [c.aps_first, c.aps_last]
        check_against_corpus_plan([c], plan, "tune")
        gts = gt_for("mvsec_tune")
        t0, t1 = int(aps[c.aps_first + warmup]), int(aps[c.aps_last])
        for stream in tune_streams:
            if stream == "real":
                store = real_store
            elif stream == "v2e":
                store = v2e_root / "tune" / f"tune_{SEQUENCE}.h5"
            else:
                raise ValueError(f"unknown tune stream {stream!r}")
            store_entry(store)
            cd = [ClipData(c, [EventArrays.from_store(store, t0, t1)])]
            windows = thin(build_windows(cd, aps, dts, warmup, gts), max_tune_windows)
            tune[stream] = WindowDataset(cd, windows, gts, augment=None, hood_row=None)
    return MvsecData(train=train, tune=tune, provenance=prov)

