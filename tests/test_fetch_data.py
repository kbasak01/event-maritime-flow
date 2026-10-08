"""scripts/fetch_data.py without network: source config, safe extraction, skip logic."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_data", REPO / "scripts/fetch_data.py")
fetch_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_data)


def test_committed_sources_are_valid():
    sources = fetch_data.load_sources(REPO / "configs/data_sources.yaml")
    assert {s["dataset"] for s in sources} == {"mvsec", "dsec", "eraft", "smd", "lars"}


@pytest.mark.parametrize(
    "bad",
    [
        {"dataset": "x", "name": "a", "kind": "ftp", "dest": "a"},
        {"dataset": "x", "name": "a", "kind": "http", "dest": "a"},
        {"dataset": "x", "name": "a", "kind": "gdrive", "dest": "a"},
        {"dataset": "x", "name": "a", "kind": "local", "dest": "a", "extract": "zip"},
        {"dataset": "x", "name": "a", "kind": "local", "dest": "../a"},
        {"dataset": "x", "name": "a", "kind": "local", "dest": "/abs"},
    ],
)
def test_invalid_sources_rejected(bad):
    with pytest.raises(ValueError):
        fetch_data.validate_sources([bad])


def test_duplicate_dest_rejected():
    s = {"dataset": "x", "name": "a", "kind": "local", "dest": "a"}
    with pytest.raises(ValueError):
        fetch_data.validate_sources([s, dict(s, name="b")])


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, b"x")


def test_zip_traversal_detected(tmp_path):
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    _zip(good, ["a/b.txt", "c.txt"])
    _zip(bad, ["ok.txt", "../evil.txt", "/abs.txt"])
    assert fetch_data.unsafe_members(good) == []
    assert fetch_data.unsafe_members(bad) == ["../evil.txt", "/abs.txt"]


def test_tar_symlink_detected(tmp_path):
    path = tmp_path / "t.tar"
    with tarfile.open(path, "w") as t:
        data = tarfile.TarInfo("ok.txt")
        data.size = 1
        t.addfile(data, io.BytesIO(b"x"))
        link = tarfile.TarInfo("link")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        t.addfile(link)
    assert fetch_data.unsafe_members(path) == ["link"]


def test_extract_is_idempotent_and_can_delete_archive(tmp_path):
    _zip(tmp_path / "a.zip", ["d/f.txt"])
    src = {"name": "a", "dest": "a.zip", "extract": "zip", "extract_to": "out",
           "delete_archive": True}
    assert fetch_data.extract(src, tmp_path) == "extracted"
    assert (tmp_path / "out/d/f.txt").read_bytes() == b"x"
    assert not (tmp_path / "a.zip").exists()
    assert fetch_data.extract(src, tmp_path) == "extracted-earlier"
    src["kind"] = "http"
    assert fetch_data.fetch(src, tmp_path, downloader=pytest.fail) == "extracted-earlier"


def test_fetch_skips_complete_and_checks_size(tmp_path):
    (tmp_path / "have.bin").write_bytes(b"abc")
    src = {"name": "h", "kind": "http", "url": "u", "dest": "have.bin", "expected_bytes": 3}
    assert fetch_data.fetch(src, tmp_path, downloader=pytest.fail) == "present"

    def fake(s, part):
        part.write_bytes(b"ab")

    short = dict(src, dest="new.bin")
    with pytest.raises(fetch_data.FetchError):
        fetch_data.fetch(short, tmp_path, downloader=fake)
    ok = dict(short, expected_bytes=2)
    assert fetch_data.fetch(ok, tmp_path, downloader=fake) == "downloaded"
    assert (tmp_path / "new.bin").read_bytes() == b"ab"


def test_gdrive_html_page_rejected(tmp_path, monkeypatch):
    def fake_curl(url, part):
        part.write_bytes(b"<!DOCTYPE html>quota")

    monkeypatch.setattr(fetch_data, "curl", fake_curl)
    src = {"name": "g", "kind": "gdrive", "gdrive_id": "abc", "resourcekey": "0-k", "dest": "g.bin"}
    with pytest.raises(fetch_data.FetchError):
        fetch_data.fetch(src, tmp_path)
    assert not list(tmp_path.iterdir())
    assert fetch_data.gdrive_url(src).endswith("id=abc&export=download&confirm=t&resourcekey=0-k")
