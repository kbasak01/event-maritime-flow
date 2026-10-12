"""Phase 7 parts of ``results/TABLES.md`` (called by ``scripts/make_tables.py``; D-057, D-058).

- ``table_e``: maritime arms on real ``mvsec_test`` — **T1**, rendered inside the T1/T2 part.
- ``t3_part``: the separate "Maritime tables (T3)" part: F-T3a (teacher-EPE family vs the RAFT
  teacher, with mask coverage), G-T3b (FWL, RSAT), H (sea-clutter event budget, synthetic).
  Every row carries its tier; a T1/T2 record here, or a T3 record in ``table_e``, is an error.

Reads committed JSON only: ``results/eval_maritime/summary.json``, ``results/maritime/
clutter.json``, ``results/eval/<maritime arm>/1337/*.json``, ``results/eval/aggregate.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from evflow.eval import study
from evflow.eval import study_maritime as sm

DIG = {"teacher_epe": 3, "teacher_npe_1": 1, "teacher_npe_3": 1, "teacher_ae_deg": 1,
       "fwl": 3, "rsat": 3, "epe": 3, "teacher_mask_coverage": 3, "events_px_fraction": 3}
LAB = {"teacher_epe": "teacher-EPE (px per frame interval)",
       "teacher_npe_3": "teacher-3PE (% px with teacher-EPE > 3 px)",
       "fwl": "FWL (> 1 better)", "rsat": "RSAT (< 1 better)"}
T3_ROWS = (("zero_flow", "M0a", "–"), ("cm_baseline", "M0b", "–"),
           ("eraft_dsec", "R1", "–")) + tuple(
    (a, "M1", "/".join(map(str, study.CORE_SEEDS)) if a in study.CORE_ARMS else "1337")
    for a in study.M1_ARMS + sm.MARITIME_ARMS)
COLS = (("smd_test", "all", "SMD test (3 videos)"), ("lars_eval_kf", "all", "LaRS kf: all"),
        ("lars_eval_kf", "water", "kf: water"), ("lars_eval_kf", "sky", "kf: sky"),
        ("lars_eval_kf", "obstacle", "kf: obstacle"),
        ("lars_eval_all", "all", "LaRS all windows"))
TAG = {"eraft_dsec": " `reference_external`"}


class TierError(ValueError):
    pass


def _f(v, k):
    return "—" if v is None else f"{v:.{DIG.get(k, 3)}f}"


def _table(head, rows):
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def _cell(e, k):
    if e is None:
        return "—"
    s = f"{_f(e['mean'], k)} [{_f(e['ci95'][0], k)}, {_f(e['ci95'][1], k)}]"
    if e.get("seed_range"):
        s += f" r [{_f(e['seed_range'][0], k)}, {_f(e['seed_range'][1], k)}]"
    return s


def table_e(root: Path, sources: set) -> list[str]:
    """T1: maritime arms on real mvsec_test beside zero flow and the v2e arms (does maritime
    training hurt T1?). Unit JSONs of the maritime runs; M0a and the v2e arms from Phase 6."""
    agg = json.loads((root / "results/eval/aggregate.json").read_text())
    idx = {(e["tier"], e["arm"], e["representation"], e["sequence"], e.get("dt")): e
           for e in agg["entries"] if e["dataset"] == "mvsec"}
    runs = {r["arm"]: r for r in sm.maritime_runs(root)}
    L = ["## Table E — maritime arms on MVSEC test, real test events (T1)", "",
         "Phase 7 step 4: the two maritime arms (trained on v2e(SMD + LaRS train), selected on "
         "v2e(mvsec_tune) at dt 1, D-056) evaluated with the Table A protocol on real "
         "`mvsec_test` events and real GT; M0a and two v2e arms repeated from Table A-T1 for "
         "reference. EPE (px over the window), mean [95 % CI]; 3-seed rows `r` = range of the "
         "per-seed means.", ""]
    for dt in study.DTS:
        body = []
        for arm, model in (("zero_flow", "M0a"), ("v2e_fixed_sup", "M1"),
                           ("v2e_selfsup", "M1"), *((a, "M1") for a in sm.MARITIME_ARMS)):
            line = ["T1", arm, model]
            for s in study.MVSEC_SEQS:
                if arm == "zero_flow":
                    rel = study.baseline_path(arm, study.stem("mvsec", s, dt))
                elif arm in sm.MARITIME_ARMS:
                    if arm not in runs:
                        line.append("— (run not trained)")
                        continue
                    rel = study.m1_path(arm, sm.MARITIME_SEED, study.stem("mvsec", s, dt))
                else:
                    e = idx.get(("T1", arm, study.MAIN_REP, s, dt))
                    if e is None:
                        line.append("—")
                        continue
                    r = (e.get("seed_range") or {}).get("epe")
                    line.append(_f(e["metrics"]["epe"], "epe") + " [" + ", ".join(
                        _f(v, "epe") for v in e["ci95"]["epe"]) + "]" + (
                        f" r [{_f(r[0], 'epe')}, {_f(r[1], 'epe')}]" if r and len(
                            e["seeds"]) > 1 else ""))
                    sources.add("results/eval/aggregate.json")
                    continue
                p = root / rel
                if not p.exists():
                    line.append("— (expected, not present)")
                    continue
                d = json.loads(p.read_text())
                if d["tier"] != "T1":
                    raise TierError(f"{rel}: tier {d['tier']} in a T1 table")
                line.append(f"{_f(d['metrics']['epe'], 'epe')} [" + ", ".join(
                    _f(v, "epe") for v in d["ci95"]["epe"]) + "]")
                sources.add(str(Path(rel).parent) + "/")
            body.append(line)
        L += [f"**EPE (px), dt = {dt}**", ""]
        L += _table(["tier", "arm", "model"] + list(study.MVSEC_SEQS), body) + [""]
    return L


def _entries(summary: dict) -> dict:
    out = {}
    for e in summary["entries"]:
        if not str(e["tier"]).startswith("T3"):
            raise TierError(f"results/eval_maritime/summary.json: tier {e['tier']} in the T3 "
                            "part")
        if e["metric"] == "epe":
            raise TierError("T3 entry with metric 'epe' (must be teacher_epe)")
        out[(e["tier"], e["arm"], e["unit"], e["slice"], e["metric"])] = e
    return out


def t3_part(root: Path, sources: set) -> list[str]:
    sp = root / "results/eval_maritime/summary.json"
    cp = root / "results/maritime/clutter.json"
    L = ["# Maritime tables (T3)", "",
         "**Evidence tiers.** T3a = v2e events simulated from maritime video (SMD on-board, "
         "LaRS), scored against a frame-based RAFT teacher on the source frames "
         "(\"teacher-EPE\": agreement with the teacher, not accuracy; the teacher is not ground "
         "truth). T3b = the same events, self-consistency only (FWL, RSAT; not accuracy). No "
         "T1/T2 row appears in this part, and no T3 row appears above it. v2e events inherit "
         "the source video's dynamic range, blur and frame rate, so nothing here measures "
         "glare or dynamic-range behaviour of a real event sensor (plan §0(1)).", "",
         "**Protocol (D-054, D-057).** 640×360 v2e output grid (LaRS sources of other aspect "
         "ratios are stretched); windows of one source-frame interval, flow in px per frame "
         "interval; SMD at 30 fps (ffprobe), **LaRS at an assumed 30 fps (D-024)** — px per "
         "frame interval does not depend on it; the first windows of each v2e unit are dropped "
         "(warm-up, count in the summary's `protocol`). LaRS kf = the window ending at each "
         "keyframe of the `lars_eval` corpus clips, sliced by the keyframe's LaRS label "
         "(water / sky / obstacle; all = full frame). Teacher mask = "
         "forward-backward-consistent ∧ pixel has events ∧ slice (D-055). mean [95 % CI]: SMD "
         "pooled = mean of the three video means, stratified moving-block bootstrap (1 s "
         "blocks); LaRS = cluster bootstrap over clips; 1000 resamples, seed 0. 3-seed arms: "
         "per-window seed mean, `r` = range of the per-seed means. M1 arms other than the "
         "maritime ones were trained on MVSEC driving (346×260) and are applied zero-shot. "
         "R1 = E-RAFT DSEC checkpoint, `reference_external` (trained on 100 ms DSEC volumes).",
         ""]
    if not sp.exists():
        return L + ["T3 evaluation summary not present yet (`results/eval_maritime/"
                    "summary.json`).", ""]
    summary = json.loads(sp.read_text())
    sources.add("results/eval_maritime/summary.json")
    ent = _entries(summary)
    head = ["tier", "arm", "model", "seeds"] + [c[2] for c in COLS]
    for tier, title, metrics in (
            ("T3a", "Table F-T3a — agreement with the RAFT teacher (T3a)",
             ("teacher_epe", "teacher_npe_3")),
            ("T3b", "Table G-T3b — motion-compensation self-consistency (T3b)", ("fwl", "rsat"))):
        L += [f"## {title}", ""]
        if tier == "T3a":
            t = summary.get("teacher", {})
            rows = []
            for k, lab in (("teacher_mask_coverage", "teacher mask coverage (FB-consistent px "
                                                     "/ slice px)"),
                           ("events_px_fraction", "px with events / slice px"),
                           ("mean_slice_px_fraction", "slice px / frame px")):
                rows.append(["T3a", lab] + [_f((t.get(u, {}).get(s) or {}).get(k), k)
                                           for u, s, _ in COLS])
            L += ["**Teacher mask and event coverage** (model-independent; mean over windows)",
                  ""] + _table(["tier", "quantity"] + [c[2] for c in COLS], rows) + [""]
        for k in metrics:
            body = []
            for arm, model, seeds in T3_ROWS:
                body.append([tier, arm + TAG.get(arm, ""), model, seeds] + [
                    _cell(ent.get((tier, arm, u, s, k)), k) for u, s, _ in COLS])
            L += [f"**{LAB[k]}**", ""] + _table(head, body) + [""]
    if summary.get("missing"):
        L += [f"Missing units: {len(summary['missing'])} (listed in "
              "`results/eval_maritime/summary.json`).", ""]
    L += ["## Table H — sea-clutter event budget, synthetic v2e(LaRS) events (T3b)", ""]
    if not cp.exists():
        return L + ["Clutter study not present yet (`results/maritime/clutter.json`).", ""]
    c = json.loads(cp.read_text())
    if c.get("tier") != "T3b":
        raise TierError("results/maritime/clutter.json: expected tier T3b")
    sources.add("results/maritime/clutter.json")
    L += [f"Keyframe windows of the {c['n_clips']} `lars_eval` clips converted at three "
          "contrast-threshold multipliers on the calibrated profile (D-054, D-058); each event "
          "takes the LaRS class of its pixel. Event share = class events / all events; pixel "
          "share = class px / all px; enrichment = event share / pixel share; events per px per "
          "frame interval (LaRS frame rate assumed, D-024). Pooled over clips (event-weighted), "
          "[95 % CI] = cluster bootstrap over clips. Synthetic: these describe v2e on LaRS "
          "frames, not a real sensor.", ""]
    rows = []
    for d in c["by_ct"].values():
        for cls in ("water", "sky", "obstacle"):
            p = d["pooled"][cls]

            def cc(k, p=p):
                v = p[k]
                return (f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
                        if v["ci95"] else f"{v['mean']:.3f}")
            rows.append(["T3b", f"×{d['ct_multiplier']:g}", cls, cc("event_share"),
                         cc("pixel_share"), cc("enrichment"), cc("events_per_px")])
    L += _table(["tier", "CT", "class", "event share", "pixel share", "enrichment",
                 "events / px / frame interval"], rows) + [""]
    rows = []
    for name, a in c["attributes_ct100"].items():
        rho = a.get("spearman_water_enrichment")
        rtxt = ("—" if not rho or rho.get("rho") is None else
                f"{rho['rho']:.2f} [{rho['ci95'][0]:.2f}, {rho['ci95'][1]:.2f}] (n={rho['n']})")
        levels = "; ".join(
            f"{lv}: n={v['n']}" + (f", enr. {v['water_enrichment']:.2f}"
                                   if v.get("reported", True) else " (not reported)")
            for lv, v in a["levels"].items())
        rows.append(["T3b", name, levels, rtxt])
    L += ["**Water enrichment by LaRS scene attribute (CT ×1.0)** — levels with fewer than 3 "
          "clips are not reported; ρ = Spearman of water enrichment on the ordinal / binary "
          "attribute, clip-bootstrap 95 % CI.", ""]
    L += _table(["tier", "attribute", "levels (n clips, mean water enrichment)",
                 "Spearman ρ [CI]"], rows) + [""]
    return L
