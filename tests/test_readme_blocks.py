"""scripts/readme_blocks.py: README numbers live only in generated blocks (CLAUDE.md rule 3,
D-067). Block replacement is strict and idempotent, section extraction follows TABLES.md
headings, and make_tables.py --check catches a stale or hand-edited README block."""

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load("readme_blocks")


def _doc(*names: str, body: str = "old") -> str:
    parts = ["# Title", "", "prose without numbers", ""]
    for n in names:
        parts += [rb.BEGIN.format(name=n), body, rb.END.format(name=n), ""]
    return "\n".join(parts)


def test_replace_and_idempotent():
    text = _doc("a", "b")
    out = rb.replace_blocks(text, {"a": "| x |\n|---|\n| 1.000 |", "b": "two"})
    assert "| 1.000 |" in out and "two" in out and "old" not in out
    assert out.startswith("# Title\n\nprose without numbers")
    assert rb.replace_blocks(out, {"a": "| x |\n|---|\n| 1.000 |", "b": "two"}) == out


def test_unknown_block_in_readme_is_an_error():
    with pytest.raises(rb.BlockError, match="no generator"):
        rb.replace_blocks(_doc("a", "zzz"), {"a": "x"})


def test_generated_block_missing_from_readme_is_an_error():
    with pytest.raises(rb.BlockError, match="not in the README"):
        rb.replace_blocks(_doc("a"), {"a": "x", "b": "y"})


def test_unbalanced_and_duplicate_markers():
    with pytest.raises(rb.BlockError, match="unbalanced"):
        rb.replace_blocks("x\n" + rb.BEGIN.format(name="a") + "\nbody\n", {"a": "x"})
    with pytest.raises(rb.BlockError, match="more than once"):
        rb.replace_blocks(_doc("a", "a"), {"a": "x"})


def test_section_and_chunk():
    md = "\n".join(["# Part", "", "## Table X — t", "", "intro", "", "**EPE, dt = 1**", "",
                    "| a |", "|---|", "| 1 |", "", "**EPE, dt = 4**", "", "| a |", "|---|",
                    "| 4 |", "", "### Sub", "", "sub body", "", "## Table Y", "", "y"])
    sec = rb.section(md, "## Table X")
    assert sec[0] == "intro" and "y" not in sec and "sub body" in sec
    assert rb.section(md, "### Sub") == ["sub body"]
    assert rb.chunk(sec, "**EPE, dt = 4**") == ["**EPE, dt = 4**", "", "| a |", "|---|",
                                                "| 4 |"]
    with pytest.raises(rb.BlockError):
        rb.section(md, "## Table Z")
    with pytest.raises(rb.BlockError):
        rb.chunk(sec, "**AE**")


def test_repo_blocks_render_and_carry_tiers():
    mt = _load("make_tables")
    blocks = rb.render_blocks(REPO, mt.render(REPO))
    assert set(blocks) == set(rb.BLOCKS)
    for name in ("table-b-t1", "table-a-t1"):
        rows = [ln for ln in blocks[name].splitlines() if ln.startswith("| T")]
        assert rows and all(ln.startswith("| T1 |") for ln in rows)
    assert all(ln.startswith("| T2 |") for ln in blocks["table-b-t2"].splitlines()
               if ln.startswith("| T"))
    for name in ("table-f-t3a", "table-g-t3b", "table-h-clutter"):
        rows = [ln for ln in blocks[name].splitlines() if ln.startswith("| T")]
        assert rows and all(ln.startswith("| T3") for ln in rows), name
    kf = blocks["key-findings"]
    assert "T1" in kf and "T3b" in kf and "T2" in kf
    assert "accuracy" not in kf.lower()
    for word in ("HDR", "high dynamic range", "low latency", "microsecond", "novel",
                 "state of the art", "significant", "guarantee"):
        assert word.lower() not in kf.lower(), word


def test_check_detects_stale_readme(tmp_path, capsys):
    mt = _load("make_tables")
    if not (REPO / "README.md").exists():
        pytest.skip("README.md not written yet")
    for rel in ("results", "configs"):
        shutil.copytree(REPO / rel, tmp_path / rel,
                        ignore=shutil.ignore_patterns("*.npz", "*.png", "*.gif", "figures"))
    shutil.copy(REPO / "README.md", tmp_path / "README.md")
    assert mt.main(["--check", "--root", str(tmp_path)]) == 0
    p = tmp_path / "README.md"
    begin = rb.BEGIN.format(name="table-b-t1")
    text = p.read_text()
    i = text.index(begin) + len(begin)
    p.write_text(text[:i] + "\n| T1 | hand-typed | 9.999 |" + text[i:])
    assert mt.main(["--check", "--root", str(tmp_path)]) == 1
    assert "README.md is stale" in capsys.readouterr().err
