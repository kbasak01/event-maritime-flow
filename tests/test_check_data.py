"""scripts/check_data.py on a fake dataset tree (no real data)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_data", REPO / "scripts/check_data.py")
check_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_data)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    lars = tmp_path / "lars"
    (lars / "sub").mkdir(parents=True)
    (lars / "small.zip").write_bytes(b"annotations")
    (lars / "small.zip:Zone.Identifier").write_bytes(b"[ZoneTransfer]")
    (lars / "sub" / "big.bin").write_bytes(b"x" * 2048)
    cfg = {
        "lars": {"root": str(lars)},
        "mvsec": {"root": str(tmp_path / "missing")},
        "v2e": {
            "repo": str(tmp_path),
            "python": "/usr/bin/python3",
            "slomo_ckpt": str(lars / "small.zip"),
        },
        "outputs": {"events": str(tmp_path / "out")},
    }
    return tmp_path, cfg


def test_inventory_contents(tree):
    tmp_path, cfg = tree
    inv = check_data.build_inventory(cfg, sha_max_bytes=1024, max_files=100)
    assert inv["kind"] == "data_inventory"
    assert set(inv["datasets"]) == {"lars.root", "mvsec.root", "v2e.slomo_ckpt"}
    assert inv["summary"]["absent"] == ["mvsec.root"]

    lars = inv["datasets"]["lars.root"]
    assert lars["path"] == "~/lars"
    assert lars["n_files"] == 2 and lars["total_bytes"] == len(b"annotations") + 2048
    by_rel = {f["rel"]: f for f in lars["files"]}
    assert set(by_rel) == {"small.zip", "sub/big.bin"}
    assert by_rel["small.zip"]["sha256"] == hashlib.sha256(b"annotations").hexdigest()
    assert by_rel["sub/big.bin"]["sha256"] is None

    ckpt = inv["datasets"]["v2e.slomo_ckpt"]
    assert ckpt["kind"] == "file" and ckpt["files"][0]["rel"] == "small.zip"


def test_listing_cap_counts_the_rest(tree):
    _, cfg = tree
    lars = check_data.build_inventory(cfg, sha_max_bytes=1024, max_files=1)["datasets"]["lars.root"]
    assert lars["files_listed"] == 1
    assert lars["unlisted_counts_by_top_dir"] == {"sub": 1}


def test_main_writes_json(tree):
    tmp_path, cfg = tree
    paths = tmp_path / "paths.yaml"
    paths.write_text(json.dumps(cfg))  # JSON is valid YAML
    out = tmp_path / "results" / "inv.json"
    assert check_data.main(["--paths", str(paths), "--out", str(out)]) == 0
    mvsec = json.loads(out.read_text())["datasets"]["mvsec.root"]
    assert mvsec == {"path": "~/missing", "present": False}
