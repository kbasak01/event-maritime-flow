"""scripts/make_release.py: only weights, graphs and text records can be staged (checklist G2),
manifest paths are made repo-relative, and SHA256SUMS.txt is in `sha256sum -c` format (D-066)."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_release", REPO / "scripts/make_release.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


@pytest.mark.parametrize("name", ["x.h5", "x.hdf5", "x.aedat4", "x.npz", "x.npy", "x.png",
                                  "x.gif", "x.mp4", "x.zip", "x.engine"])
def test_data_and_media_refused(name):
    with pytest.raises(mr.ReleaseError):
        mr.check_asset_name(name)


@pytest.mark.parametrize("name", ["m1_a.pt", "m1_a_fp32.onnx", "run_manifests.json",
                                  "WEIGHTS_LICENSE.md", "SHA256SUMS.txt"])
def test_assets_allowed(name):
    mr.check_asset_name(name)


def test_relativise():
    repo, home = Path("/h/u/repo"), Path("/h/u")
    doc = {"a": "/h/u/repo/src/x.py", "b": ["/h/u/repo", "/h/u/other/y"], "c": 3}
    assert mr.relativise(doc, repo, home) == {"a": "src/x.py", "b": [".", "~/other/y"], "c": 3}


def test_sums_text(tmp_path):
    (tmp_path / "b.pt").write_bytes(b"bb")
    (tmp_path / "a.onnx").write_bytes(b"a")
    txt = mr.sums_text(sorted(tmp_path.iterdir()))
    assert txt == (f"{hashlib.sha256(b'a').hexdigest()}  a.onnx\n"
                   f"{hashlib.sha256(b'bb').hexdigest()}  b.pt\n")
