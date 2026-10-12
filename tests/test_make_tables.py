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
              "sequence": "indoor_flying1", "dt": 1, "estimate": 0.25, "ci95": [0.1, 0.4],
              "n_windows": {"indoor_flying1": 2193},
              "seed_pairings": {"range": [-0.05, 0.27], "n": 9}},
             {"table": "B", "tier": "T1", "comparison": "closure_v2e_rand_sup",
              "sequence": "indoor_flying1", "dt": 4, "estimate": 0.6, "ci95": [0.4, 1.06],
              "seed_pairings": {"range": [-0.2, 2.7], "n": 27}, "seed_unstable": True},
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
    assert "1.500 [1.400, 1.600] r [1.300, 1.700]" in a1          # 3-seed: mean [CI] range
    assert "1.200 [1.100, 1.300]" in a2 and "(n=2190)" not in a1
    assert "1.280 [1.180, 1.380]" in a1 and "1.100 [1.000, 1.200]" in a2
    assert "0.250 [0.100, 0.400]" in _section(text, "## Table B")
    b1 = _section(text, "### Table B-T1")
    assert "indoor_flying1 (n=2193)" in b1
    assert "0.250 [0.100, 0.400] r [-0.050, 0.270]" in b1             # seed range beside the CI
    assert "0.600 [0.400, 1.060] r [-0.200, 2.700] (" in b1 and "seed-unstable (D-053)" in b1
    assert "**CIs resample windows only**" in text
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


# --- Phase 7: Table E (T1) and the separate T3 part (scripts/tables_maritime.py) -------------

def _t3(root, *, tier="T3a", metric="teacher_epe"):
    e = {"tier": tier, "arm": "zero_flow", "model": "M0a", "seeds": [None], "unit": "lars_eval_kf",
         "slice": "water", "metric": metric, "mean": 2.25, "ci95": [2.0, 2.5],
         "seed_range": None, "n_windows": 40}
    _w(root, "results/eval_maritime/summary.json",
       {"kind": "maritime_eval_summary", "entries": [e], "missing": [],
        "teacher": {"lars_eval_kf": {"water": {"teacher_mask_coverage": 0.4,
                                               "events_px_fraction": 0.3,
                                               "mean_slice_px_fraction": 0.5}}}})
    by_ct = {f"ct{k:03d}": {"ct_multiplier": k / 100, "pooled": {
        c: {q: {"mean": 0.5, "ci95": [0.4, 0.6]} for q in ("event_share", "pixel_share",
                                                          "enrichment", "events_per_px")}
        for c in ("water", "sky", "obstacle")}} for k in (67, 100, 150)}
    _w(root, "results/maritime/clutter.json",
       {"kind": "clutter_study", "tier": "T3b", "n_clips": 40, "by_ct": by_ct,
        "attributes_ct100": {"waves": {"levels": {"still": {"n": 2, "reported": False}},
                                       "spearman": {"reported": False}}}})


def test_t3_part_is_separate_and_tagged(root):
    _t3(root)
    text = mt.render(root)
    head = text.index("# Maritime tables (T3)")
    above, below = text[:head], text[head:]
    assert "| T3" not in above                                   # no T3 row in Tables A-E
    assert "| T1 |" not in below and "| T2 |" not in below        # no T1/T2 row in the T3 part
    f = _section(text, "## Table F-T3a")
    assert "2.250 [2.000, 2.500]" in f and "teacher-EPE" in f and "| EPE" not in f
    assert "0.400" in f                                          # coverage row
    assert "assumed 30 fps (D-024)" in below
    assert "## Table H" in below and "×0.67" in below
    assert "## Table E" in above and text.index("## Table E") < head


def test_t3_part_refuses_wrong_tiers_and_epe(root):
    _t3(root, tier="T1")
    with pytest.raises(mt.tables_maritime.TierError):
        mt.render(root)
    _t3(root, metric="epe")
    with pytest.raises(mt.tables_maritime.TierError):
        mt.render(root)


def test_table_e_refuses_a_t3_unit(root):
    _w(root, "results/train/maritime_summary.json",
       {"runs": [{"run_id": "maritime_teacher_voxel5_s1337", "arm": "maritime_teacher",
                  "seed": 1337, "representation": "voxel5", "status": "done"}]})
    met, ci = _m(0.9)
    rel = study.m1_path("maritime_teacher", 1337, study.stem("mvsec", "indoor_flying1", 1))
    _w(root, rel, {"tier": "T1", "metrics": met, "ci95": ci, "n_windows": 9})
    assert "0.900 [0.800, 1.000]" in _section(mt.render(root), "## Table E")
    _w(root, rel, {"tier": "T3a", "metrics": met, "ci95": ci, "n_windows": 9})
    with pytest.raises(mt.tables_maritime.TierError):
        mt.render(root)
