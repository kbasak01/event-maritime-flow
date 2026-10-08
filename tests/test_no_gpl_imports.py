"""No GPL code in this MIT repo (CLAUDE.md rule 6, VALIDATION_CHECKLIST G1).

Scans every .py file under src/, tests/ and scripts/ for imports of the GPL-3.0
tub-rip/event_based_optical_flow package and for GPL licence text pasted into a file.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "tests", "scripts")
THIS_FILE = Path(__file__).resolve()

# Top-level module names of the GPL package. Its own code lives in a top-level `src` package, which
# this repo never imports (our package is `evflow`), so `import src...` is treated as a leak too.
GPL_MODULES = {"event_based_optical_flow", "src"}
GPL_TEXT = (
    "tub-rip/event_based_optical_flow",
    "GNU General Public License",
    "SPDX-License-Identifier: GPL",
    "GPL-3.0",
)


def gpl_findings(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [f"{path}: text '{t}'" for t in GPL_TEXT if t in text]
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        bad = [n for n in names if n.split(".")[0] in GPL_MODULES]
        found += [f"{path}:{node.lineno}: import {n}" for n in bad]
    return found


def py_files() -> list[Path]:
    files = [p for d in SCAN_DIRS for p in (REPO / d).rglob("*.py")]
    return [p for p in files if p.resolve() != THIS_FILE]


def test_scan_covers_package():
    assert any(p.parts[-2:] == ("evflow", "__init__.py") for p in py_files())


def test_no_gpl_imports_or_text():
    findings = [f for p in py_files() for f in gpl_findings(p)]
    assert not findings, "\n".join(findings)


def test_scanner_detects_planted_import(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from event_based_optical_flow.costs import foo\nimport src.solver\n")
    assert len(gpl_findings(bad)) == 2


def test_scanner_detects_licence_text(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("# This program is free software: GNU General Public License v3\n")
    assert gpl_findings(bad)
