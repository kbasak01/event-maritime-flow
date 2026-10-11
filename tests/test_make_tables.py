"""scripts/make_tables.py: renders from committed JSON only, keeps tiers apart, refuses T3 rows,
footnotes missing cells, and --check catches a stale or hand-edited TABLES.md (checklist E2, G6)."""

import importlib.util
import json
from pathlib import Path

import pytest

from evflow.eval import study

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts/make_tables.py")
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)


def _w(root: Path, rel: str, doc: dict) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def _m(epe):
    keys = ("epe", "outlier_pct", "npe_1", "npe_2", "npe_3", "ae_deg", "fwl")
    return {k: epe for k in keys}, {k: [epe - 0.1, epe + 0.1] for k in keys}


@pytest.fixture
def root(tmp_path):
    entries = []
    for tier in ("T1", "T2"):
        for arm in study.M1_ARMS:
            seeds = list(study.expected_seeds(arm, study.MAIN_REP))
            met, ci = _m(1.5 if tier == "T1" else 1.2)
            entries.append({"tier": tier, "arm": arm, "representation": "voxel5",
                            "dataset": "mvsec", "sequence": "indoor_flying1", "dt": 1,
                            "seeds": seeds, "seeds_complete": True, "metrics": met, "ci95": ci,
                            "seed_range": {k: [v - 0.2, v + 0.2] for k, v in met.items()},
                            "per_seed": {}, "n_windows": {k: 2193 for k in met}})
    _w(tmp_path, "results/eval/aggregate.json", {"kind": "x", "entries": entries})
    pairs = [{"table": "B", "tier": "T1", "comparison": "gap_v2e_fixed_sup",
              "sequence": "indoor_flying1", "dt": 1, "estimate": 0.25, "ci95": [0.1, 0.4]},
             {"table": "B", "tier": "T1", "comparison": "closure_v2e_rand_sup",
              "sequence": "indoor_flying1", "dt": 1, "estimate": None, "ci95": None,
              "undefined_reason": "gap CI includes 0"}]
    _w(tmp_path, "results/eval/paired.json", {"kind": "x", "bootstrap": {}, "entries": pairs})
    _w(tmp_path, "results/eval/summary.json", {"kind": "x", "missing": [], "n_present": 1,
                                               "n_expected": 2})
    met, ci = _m(1.28)
    _w(tmp_path, "results/baselines/zero_flow/mvsec_indoor_flying1_dt1.json",
       {"tier": "T1", "metrics": met, "ci95": ci, "n_windows": 2193})
    met, ci = _m(1.10)
    _w(tmp_path, "results/baselines/zero_flow/mvsec_indoor_flying1_dt1_v2e.json",
       {"tier": "T2", "metrics": met, "ci95": ci, "n_windows": 2190})
    return tmp_path


def _section(text: str, title: str) -> str:
    start = text.index(title)
    nxt = text.find("\n## ", start + 1)
    return text[start: nxt if nxt > 0 else None]


def test_render_tiers_apart_and_values(root):
    text = mt.render(root)
    a1, a2 = _section(text, "## Table A-T1"), _section(text, "## Table A-T2")
    rows1 = [ln for ln in a1.splitlines() if ln.startswith("| T")]
    rows2 = [ln for ln in a2.splitlines() if ln.startswith("| T")]
    assert rows1 and all(ln.startswith("| T1 |") for ln in rows1)
    assert rows2 and all(ln.startswith("| T2 |") for ln in rows2)
    assert "1.500 [1.400, 1.600] r 1.300–1.700" in a1          # 3-seed: mean [CI] range
    assert "1.200 [1.100, 1.300]" in a2 and "(n=2190)" not in a1
    assert "1.280 [1.180, 1.380]" in a1 and "1.100 [1.000, 1.200]" in a2
    assert "0.250 [0.100, 0.400]" in _section(text, "## Table B")
    assert "undefined: gap CI includes 0" in _section(text, "## Table B")


def test_missing_cells_are_footnoted(root):
    text = mt.render(root)
    a1 = _section(text, "## Table A-T1")
    assert "— (1)" in a1 and "(1) expected, not present" in a1
    assert "cm_baseline" in a1                                  # absent baseline -> row kept


def test_t3_record_is_refused(root):
    doc = json.loads((root / "results/eval/aggregate.json").read_text())
    doc["entries"][0]["tier"] = "T3a"
    (root / "results/eval/aggregate.json").write_text(json.dumps(doc))
    with pytest.raises(mt.TierError):
        mt.render(root)


def test_untagged_record_is_refused(root):
    doc = json.loads((root / "results/eval/paired.json").read_text())
    del doc["entries"][0]["tier"]
    (root / "results/eval/paired.json").write_text(json.dumps(doc))
    with pytest.raises(mt.TierError):
        mt.render(root)


def test_check_detects_hand_edit(root, capsys):
    assert mt.main(["--root", str(root)]) == 0
    assert mt.main(["--root", str(root), "--check"]) == 0
    out = root / "results/TABLES.md"
    out.write_text(out.read_text().replace("1.500", "1.499", 1))
    assert mt.main(["--root", str(root), "--check"]) == 1


def test_repo_tables_render():
    if not (REPO / "results/eval/aggregate.json").exists():
        pytest.skip("Phase 6 summaries not generated yet")
    text = mt.render(REPO)
    assert "## Table A-T1" in text and "## Table D" in text
