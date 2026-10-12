"""Per-unit teacher store (Phase 7 step 2): ``<events>/teacher/<corpus>/<unit>.h5``.

Datasets: ``flow`` (N, 2, H, W) float32 forward teacher flow of window k (source frames k → k+1,
px per frame interval), ``valid`` (N, H, W) bool forward–backward-consistent pixels (D-055),
``valid_literal`` (N, H, W) bool (§1.6 as printed; coverage only), ``frame_index`` (N,) = k.
Attrs: unit key, ``frames_sha256`` (= the v2e unit's input SHA, so teacher and events come from
the same frames), model/weights/iterations, alphas, width/height, created_by/at, git_sha.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


def write(path: str | PathLike, out: dict, attrs: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    comp = hdf5plugin.Blosc(cname="lz4", clevel=5)
    n, _, h, w = out["flow"].shape
    with h5py.File(tmp, "w") as f:
        f.create_dataset("flow", data=out["flow"].astype(np.float32),
                         chunks=(1, 2, h, w), **comp)
        f.create_dataset("valid", data=out["valid"].astype(bool), chunks=(1, h, w), **comp)
        f.create_dataset("valid_literal", data=out["valid_literal"].astype(bool),
                         chunks=(1, h, w), **comp)
        f.create_dataset("frame_index", data=np.arange(n, dtype=np.int64))
        for k, v in {**attrs, "width": w, "height": h, "n_windows": n}.items():
            f.attrs[k] = v
    tmp.replace(path)


class TeacherCache:
    """Read access by window index k (frames k → k+1); ``.get(k) -> (flow, valid)``."""

    def __init__(self, path: str | PathLike):
        self.path = Path(path)
        self._f = h5py.File(self.path, "r")
        self.attrs = dict(self._f.attrs)
        self.n = int(self._f["flow"].shape[0])

    def get(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._f["flow"][k], self._f["valid"][k]

    def get_literal(self, k: int) -> np.ndarray:
        return self._f["valid_literal"][k]

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
