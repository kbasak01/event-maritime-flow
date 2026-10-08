"""Print the layout of an HDF5 file (stand-in for ``h5ls -rv``, which is not installed here).

Usage: python scripts/h5_layout.py FILE.h5 [--head N]
Lists every group/dataset with shape, dtype, chunking, compression filter ids and attrs; with
``--head N`` also the first N rows of each dataset with at most 1e8 elements per row block.
"""

from __future__ import annotations

import argparse

import h5py
import hdf5plugin  # noqa: F401  (registers blosc etc. so compressed datasets are readable)
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--head", type=int, default=0)
    args = ap.parse_args()

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            filters = obj.id.get_create_plist()
            fids = [filters.get_filter(i)[0] for i in range(filters.get_nfilters())]
            print(f"{name:40s} shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks} "
                  f"filters={fids} attrs={dict(obj.attrs)}")
            if args.head and obj.ndim >= 1 and obj.shape[0] > 0:
                with np.printoptions(precision=6, suppress=False, linewidth=120):
                    print("   head:", obj[: args.head])
            elif args.head and obj.ndim == 0:
                print("   value:", obj[()])
        else:
            print(f"{name}/ attrs={dict(obj.attrs)}")

    with h5py.File(args.file, "r") as f:
        print(f"{args.file}: root attrs={dict(f.attrs)}")
        f.visititems(show)


if __name__ == "__main__":
    main()
