"""Every results JSON that carries metrics carries an evidence tier and an arm (CLAUDE.md rule 1).

A results/**/*.json file is either a metric summary (has `metrics`) or a non-metric artefact that
declares its `kind` (e.g. the data inventory). Anything else fails, so untagged numbers cannot slip
in.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TIERS = {"T1", "T2", "T3a", "T3b"}


def tier_problems(path: Path) -> list[str]:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON ({e})"]
    if not isinstance(doc, dict):
        return [f"{path}: top level must be an object"]
    if "metrics" not in doc:
        return [] if doc.get("kind") else [f"{path}: neither `metrics` nor `kind`"]
    problems = []
    if doc.get("tier") not in TIERS:
        problems.append(f"{path}: tier={doc.get('tier')!r} not in {sorted(TIERS)}")
    if not isinstance(doc.get("arm"), str) or not doc["arm"]:
        problems.append(f"{path}: missing `arm`")
    metrics = doc.get("metrics") or {}
    if str(doc.get("tier", "")).startswith("T3") and "epe" in metrics:
        problems.append(f"{path}: T3 results report `teacher_epe`, never `epe`")
    return problems


def test_all_results_json_tagged():
    files = sorted((REPO / "results").rglob("*.json"))
    problems = [p for f in files for p in tier_problems(f)]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("doc", "ok"),
    [
        ({"tier": "T1", "arm": "real_sup", "metrics": {"epe": 0.8}}, True),
        ({"tier": "T3a", "arm": "maritime_teacher", "metrics": {"teacher_epe": 2.0}}, True),
        ({"kind": "data_inventory"}, True),
        ({"arm": "real_sup", "metrics": {"epe": 0.8}}, False),
        ({"tier": "T4", "arm": "real_sup", "metrics": {}}, False),
        ({"tier": "T2", "metrics": {}}, False),
        ({"tier": "T3b", "arm": "x", "metrics": {"epe": 1.0}}, False),
        ({"note": "untagged"}, False),
        ([1, 2], False),
    ],
)
def test_checker_controls(tmp_path, doc, ok):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(doc))
    assert (not tier_problems(f)) is ok
