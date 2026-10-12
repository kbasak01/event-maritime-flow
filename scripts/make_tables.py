"""Render ``results/TABLES.md`` from committed results JSON (CLAUDE.md rule 3; plan Phase 6 step 2).

    python scripts/make_tables.py            # write results/TABLES.md
    python scripts/make_tables.py --check    # regenerate in memory; exit 1 on any difference

Reads only committed JSON, so it runs without data (CI, ``make reproduce-tables``):
``results/eval/{aggregate,paired,summary}.json`` (``scripts/summarise_eval.py``) and the Phase 4
baseline units ``results/baselines/<arm>/*.json``. Nothing is typed by hand; an absent cell is
rendered "—" with a footnote that says why.

Tables (evidence tiers, CLAUDE.md rule 1 — every row carries its tier; T1 and T2 rows sit in
separate sub-sections; a T3 record makes the script fail, Phase 7 adds its own section):
  A  MVSEC test: EPE and %Out per arm x sequence x dt, mean [95 % CI]; 3-seed arms: seed mean
     [CI of the seed-mean per-window series], range over seeds. A-T1 real events, A-T2 v2e events.
  B  Simulator gap and closure with paired-bootstrap CIs (T1, then T2), and T2 - T1 per model.
  C  Representation ablation (T1, seed 1337).
  D  DSEC-Flow zero-shot generalisation (T1).
  E  Maritime arms on MVSEC test (T1; Phase 7, scripts/tables_maritime.py).
Then a separate part "Maritime tables (T3)" (scripts/tables_maritime.py): F-T3a teacher-EPE
family, G-T3b FWL/RSAT, H sea-clutter event budget.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from evflow.eval import study

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables_maritime  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_REL = "results/TABLES.md"
TIERS = ("T1", "T2")
DIGITS = {"epe": 3, "outlier_pct": 2, "npe_1": 2, "npe_2": 2, "npe_3": 2, "ae_deg": 2,
          "fwl": 3}
LABEL = {"epe": "EPE (px)", "outlier_pct": "%Out", "npe_1": "1PE (%)", "npe_3": "3PE (%)",
         "ae_deg": "AE (deg)"}
MODEL = {**study.BASELINE_MODEL, **{a: "M1" for a in study.M1_ARMS}}
TAGS_NOTE = {"eraft_mvsec20": "reference_external", "eraft_dsec":
             "reference_external, reference_seen_data"}


class TierError(ValueError):
    pass


def _check_tier(rec: dict, where: str) -> str:
    t = rec.get("tier")
    if t is None:
        raise TierError(f"{where}: record without a tier")
    if str(t).startswith("T3"):
        raise TierError(f"{where}: T3 record in the T1/T2 tables (Phase 7 has its own section)")
    if t not in TIERS:
        raise TierError(f"{where}: tier {t!r} not in {TIERS}")
    return t


class Study:
    """Committed JSON, indexed for lookup."""

    def __init__(self, root: Path) -> None:
        self.root = root
        ev = root / "results" / "eval"
        self.aggregate = json.loads((ev / "aggregate.json").read_text())
        self.paired = json.loads((ev / "paired.json").read_text())
        self.summary = json.loads((ev / "summary.json").read_text())
        self.missing = {m["path"] for m in self.summary.get("missing", [])}
        self.m1 = {}
        for e in self.aggregate["entries"]:
            t = _check_tier(e, "results/eval/aggregate.json")
            self.m1[(t, e["arm"], e["representation"], e["dataset"], e["sequence"],
                     e.get("dt"))] = e
        self.pairs = {}
        for e in self.paired["entries"]:
            t = _check_tier(e, "results/eval/paired.json")
            self.pairs[(e["table"], t, e["comparison"], e["sequence"], e.get("dt"))] = e
        self.sources: set[str] = {"results/eval/aggregate.json", "results/eval/paired.json",
                                  "results/eval/summary.json"}

    def baseline(self, tier: str, arm: str, ds: str, seq: str, dt) -> dict | None:
        rel = study.baseline_path(arm, study.stem(ds, seq, dt,
                                                  events="v2e" if tier == "T2" else "real"))
        p = self.root / rel
        if not p.exists():
            return None
        doc = json.loads(p.read_text())
        if _check_tier(doc, rel) != tier:
            raise TierError(f"{rel}: tier {doc['tier']} where {tier} was expected")
        self.sources.add(f"results/baselines/{arm}/")
        return {"metrics": doc["metrics"], "ci95": doc["ci95"], "seed_range": None,
                "n": doc["n_windows"], "seeds": None, "complete": True}

    def cell(self, tier: str, arm: str, rep: str, ds: str, seq: str, dt,
             metric: str) -> tuple[dict | None, str | None]:
        """(record, None) or (None, reason)."""
        if arm in study.BASELINE_MODEL:
            r = self.baseline(tier, arm, ds, seq, dt)
            if r is None:
                return None, "expected, not present (results/eval/summary.json `missing`)"
            return r, None
        e = self.m1.get((tier, arm, rep, ds, seq, dt))
        if e is None:
            return None, "expected, not present (results/eval/summary.json `missing`)"
        r = {"metrics": e["metrics"], "ci95": e["ci95"], "seed_range": e["seed_range"],
             "n": (e["n_windows"] or {}).get(metric), "seeds": e["seeds"],
             "complete": e.get("seeds_complete", False)}
        if not r["complete"]:
            return None, f"incomplete seeds (have {e['seeds']})"
        return r, None


class Notes:
    def __init__(self) -> None:
        self.items: list[str] = []

    def ref(self, text: str) -> str:
        if text not in self.items:
            self.items.append(text)
        return f"({self.items.index(text) + 1})"

    def mark(self, text: str) -> str:
        return f"— {self.ref(text)}"

    def render(self) -> list[str]:
        return [f"({i}) {t}  " for i, t in enumerate(self.items, 1)]


def _f(v, metric: str) -> str:
    return "—" if v is None else f"{v:.{DIGITS.get(metric, 3)}f}"


def fmt_cell(r: dict, metric: str) -> str:
    m, ci = r["metrics"].get(metric), (r["ci95"] or {}).get(metric)
    if m is None:
        return "—"
    s = _f(m, metric) + (f" [{_f(ci[0], metric)}, {_f(ci[1], metric)}]" if ci else "")
    rng = (r["seed_range"] or {}).get(metric) if r["seeds"] and len(r["seeds"]) > 1 else None
    if rng:
        s += f" r [{_f(rng[0], metric)}, {_f(rng[1], metric)}]"
    return s


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _seeds(arm: str, rep: str) -> str:
    if arm in study.BASELINE_MODEL:
        return "–"
    return "/".join(str(s) for s in study.expected_seeds(arm, rep))


def metric_table(st: Study, tier: str, ds: str, seqs: tuple, dt, metric: str,
                 rows: list[tuple[str, str]], notes: Notes) -> list[str]:
    """rows = (arm, representation); one column per sequence; n windows in the header."""
    body, ns = [], {s: set() for s in seqs}
    for arm, rep in rows:
        line = [tier, arm + (f" `{TAGS_NOTE[arm]}`" if arm in TAGS_NOTE else ""), MODEL[arm],
                rep if arm not in study.BASELINE_MODEL else "–", _seeds(arm, rep)]
        for s in seqs:
            r, why = st.cell(tier, arm, rep, ds, s, dt, metric)
            if r is None:
                line.append(notes.mark(why))
                continue
            line.append(fmt_cell(r, metric))
            if r["n"] is not None:
                ns[s].add(r["n"])
        body.append(line)
    head = ["tier", "arm", "model", "rep.", "seeds"]
    for s in seqs:
        n = ns[s]
        head.append(f"{s} (n={next(iter(n))})" if len(n) == 1 else
                    (f"{s} (n={min(n)}–{max(n)})" if n else s))
    return _table(head, body)


def pair_cell(e: dict | None, notes: Notes) -> str:
    if e is None:
        return notes.mark("comparison not computable: a series is missing or has incomplete "
                          "seeds (results/eval/paired.json)")
    if e["estimate"] is None:
        return notes.mark(f"undefined: {e.get('undefined_reason', 'see paired.json')}")
    lo, hi = e["ci95"]
    s = f"{_pf(e['estimate'])} [{_pf(lo)}, {_pf(hi)}]"
    sp = e.get("seed_pairings")
    if sp and sp.get("range"):
        s += f" r [{_pf(sp['range'][0])}, {_pf(sp['range'][1])}]"
    if e.get("seed_unstable"):
        s += " " + notes.ref("seed-unstable (D-053): gap(v2e_fixed_sup) ≤ 0 for some single-seed "
                             "pairings (`n_fixed_gap_pairings_nonpositive` in paired.json); `r` "
                             "covers the pairings with a positive gap only")
    return s


def _pf(v: float) -> str:
    """3 decimals; a nonzero value that would print as ±0.000 keeps 4 decimals."""
    return f"{v:.4f}" if v != 0 and abs(v) < 0.0005 else f"{v:.3f}"


def pair_table(st: Study, table: str, tier: str, dt, comps: list[tuple[str, str]],
               notes: Notes) -> list[str]:
    """One row per comparison; per-sequence columns carry the paired window count n (windows
    scored in every series of the comparison; a range if the rows differ)."""
    cols = list(study.MVSEC_SEQS) + ["pooled"]
    body, ns = [], {c: set() for c in study.MVSEC_SEQS}
    for comp, label in comps:
        row = [tier, label]
        for c in cols:
            e = st.pairs.get((table, tier, comp, c, dt))
            row.append(pair_cell(e, notes))
            n = (e or {}).get("n_windows", {}).get(c)
            if n is not None:
                ns[c].add(n)
        body.append(row)
    head = []
    for c in cols:
        if c == "pooled":
            head.append("pooled (mean of 4)")
        elif ns[c]:
            head.append(f"{c} (n={min(ns[c])})" if len(ns[c]) == 1
                        else f"{c} (n={min(ns[c])}–{max(ns[c])})")
        else:
            head.append(c)
    return _table(["tier", "comparison"] + head, body)


def render(root: Path = REPO) -> str:
    st = Study(root)
    rep = study.MAIN_REP
    L: list[str] = []
    boot = st.paired.get("bootstrap", {})
    L += ["# Results tables (T1/T2)", "",
          "Generated by `scripts/make_tables.py` from committed JSON — do not edit by hand; "
          "`make reproduce-tables` checks that this file is current.", "",
          "**Evidence tiers.** T1 = real events, real ground-truth flow. T2 = v2e events "
          "simulated from the same sensor's APS frames (MVSEC), real ground-truth flow. "
          "Every row carries its tier; T1 and T2 rows never share a table. Maritime T3 "
          "results are in the separate part \"Maritime tables (T3)\" below; no T3 row appears "
          "in Tables A–E.", "",
          "**Statistics.** Mean over evaluation windows of the per-window mean "
          "(configs/eval_mvsec.yaml). [lo, hi] = 95 % moving-block bootstrap CI over windows "
          f"(block {boot.get('block_seconds', '?')} s of windows, "
          f"{boot.get('resamples', '?')} resamples, seed {boot.get('seed', '?')}). 3-seed "
          "arms: seed mean, CI of the seed-mean per-window series, `r` = range of the "
          "per-seed means. Gaps, closure and T2 − T1: paired bootstrap on the same windows; "
          "pooled = mean of the four sequence means, each sequence resampled independently. "
          "**CIs resample windows only** (each arm's seed mean held fixed); between-seed "
          "variation is shown by `r` (Table A: range of the per-seed means; Table B: range over "
          "single-seed combinations, D-053) and is not in the CI. "
          "EPE in px over the window (dt = k APS frame intervals; never compared across dt).", ""]

    a_rows_t1 = ([("zero_flow", ""), ("cm_baseline", ""), ("eraft_mvsec20", "")]
                 + [(a, rep) for a in study.M1_ARMS])
    a_rows_t2 = [("zero_flow", "")] + [(a, rep) for a in study.M1_ARMS]
    for tier, rows, title in (("T1", a_rows_t1, "real test events"),
                              ("T2", a_rows_t2, "v2e(APS) test events, calibrated profile")):
        notes = Notes()
        L += [f"## Table A-{tier} — MVSEC test, {title} ({tier})", ""]
        for dt in study.DTS:
            for metric in ("epe", "outlier_pct"):
                L += [f"**{LABEL[metric]}, dt = {dt}**", ""]
                L += metric_table(st, tier, "mvsec", study.MVSEC_SEQS, dt, metric, rows, notes)
                L += [""]
        L += ["Notes:  "]
        if tier == "T1":
            L += ["- `zero_flow` (M0a) is the reference every arm is read against; `real_sup` "
                  "is above it on indoor_flying1 dt = 1 (D-046).  ",
                  "- R1 = E-RAFT 20 Hz MVSEC checkpoint, `reference_external`, trained on "
                  "outdoor_day2; worse than zero flow on indoor dt = 1 (D-042).  ",
                  "- All M1 arms train and select identically on the same outdoor_day2 windows "
                  "(D-043); only the training events differ.  "]
        else:
            L += ["- Same windows, GT and region as A-T1; events and the sparse mask come from "
                  "the calibrated v2e(APS) test corpus (D-035, D-051), so n can differ from "
                  "A-T1.  ",
                  "- M0b and R1 are not evaluated on T2 events, by design (D-051).  "]
        L += notes.render() + [""]

    L += ["## Table B — simulator gap and closure (paired bootstrap)", "",
          "gap = EPE(v2e arm) − EPE(real arm) on the same windows (> 0: the v2e-trained arm "
          "is worse). closure = 1 − gap(v2e_rand_sup) / gap(v2e_fixed_sup). Supervised arms: "
          "3-seed means; self-supervised arms: one seed each. n = windows scored in every series "
          "of the comparison.", "",
          "Reading notes:  ",
          "- `r` = range of the statistic over single-seed pairings (supervised gaps: 3 × 3 "
          "seeds; closure: 3 × 3 × 3; T2 − T1: same model, 3 seeds); the CI covers windows "
          "only. A CI that excludes 0 while `r` spans 0 means the sign is not stable across "
          "training seeds.  ",
          "- Indoor dt = 1: on indoor_flying1 no supervised M1 arm has a lower EPE than zero "
          "flow, and on indoor_flying3 real_sup does not either (Table A-T1; D-046); negative "
          "gaps on these sequences compare models at or near the zero-flow floor and are not "
          "evidence that v2e training is better.  ",
          "- closure is a ratio: where gap(v2e_fixed_sup) is small its CI is wide; values above "
          "1 (randomisation overshoots the real arm) are within that uncertainty, not a "
          "finding.  ", ""]
    comps = [("gap_v2e_fixed_sup", "gap v2e_fixed_sup − real_sup"),
             ("gap_v2e_rand_sup", "gap v2e_rand_sup − real_sup"),
             ("closure_v2e_rand_sup", "closure (CT randomisation)"),
             ("gap_selfsup", "gap v2e_selfsup − real_selfsup")]
    for tier, title in (("T1", "real test events"), ("T2", "v2e(APS) test events")):
        notes = Notes()
        L += [f"### Table B-{tier} — {title} ({tier})", ""]
        for dt in study.DTS:
            L += [f"**EPE gap, dt = {dt}**", ""]
            L += pair_table(st, "B", tier, dt, comps, notes) + [""]
        if tier == "T2":
            L += ["**T2 − T1: the same model on v2e vs real test events** (EPE difference, same "
                  "windows and GT, each stream's own sparse mask; < 0: lower EPE on v2e events). "
                  "These rows combine a T2 series with the same model's T1 series; they are "
                  "tagged T2 because they measure the simulator (D-051(c)).", ""]
            tcomps = [(f"t2_minus_t1_{a}", a) for a in (*study.M1_ARMS, "zero_flow")]
            for dt in study.DTS:
                L += [f"*dt = {dt}*", ""]
                L += pair_table(st, "T1_vs_T2", "T2", dt, tcomps, notes) + [""]
        L += notes.render() + [""]

    notes = Notes()
    L += ["## Table C — representation ablation (T1, seed 1337)", "",
          "Is the simulator gap representation-dependent? real_sup and v2e_fixed_sup trained "
          "per representation with seed 1337 (D-043(f)); voxel5 cells are the seed-1337 main "
          "runs.", ""]
    c_rows = [(a, r) for r in study.ABLATION_REPS for a in study.ABLATION_ARMS]
    for dt in study.DTS:
        L += [f"**EPE (px), dt = {dt}**", ""]
        L += _c_table(st, dt, c_rows, notes) + [""]
        L += [f"**EPE gap v2e_fixed_sup − real_sup, dt = {dt}**", ""]
        L += pair_table(st, "C", "T1", dt, [(f"gap_v2e_fixed_sup_{r}", r)
                                            for r in study.ABLATION_REPS], notes) + [""]
    L += notes.render() + [""]

    notes = Notes()
    L += ["## Table D — DSEC-Flow zero-shot generalisation (T1)", "",
          "M1 arms trained at 346×260 on MVSEC outdoor_day2 and applied unchanged to DSEC's "
          "640×480 rectified frame (zero-shot; resolution, sensor and window length (100 ms) all "
          "change). Mask: valid GT over the full frame. R1 = E-RAFT DSEC checkpoint, "
          "`reference_external`, `reference_seen_data` (E-RAFT was trained on DSEC train "
          "sequences, D-040; its rows validate the pipeline, not generalisation).",
          ""]
    d_rows = ([("zero_flow", ""), ("cm_baseline", ""), ("eraft_dsec", "")]
              + [(a, rep) for a in study.M1_ARMS])
    for metric in ("epe", "npe_1", "npe_3", "ae_deg"):
        L += [f"**{LABEL[metric]}**", ""]
        L += metric_table(st, "T1", "dsec", study.DSEC_SEQS, None, metric, d_rows, notes)
        L += [""]
    L += notes.render() + [""]

    L += tables_maritime.table_e(root, st.sources)
    L += tables_maritime.t3_part(root, st.sources)
    L += ["# Sources", ""] + [f"- `{s}`" for s in sorted(st.sources)] + [
        f"- units: {st.summary.get('n_present')}/{st.summary.get('n_expected')} present "
        f"(`results/eval/summary.json`)", ""]
    return "\n".join(L).rstrip() + "\n"


def _c_table(st: Study, dt: int, rows: list[tuple[str, str]], notes: Notes) -> list[str]:
    """Seed-1337 unit JSON of every representation (the voxel5 cells are the main runs' seed
    1337, so all cells are single-seed and comparable)."""
    body, ns = [], {s: set() for s in study.MVSEC_SEQS}
    for arm, rep in rows:
        line = ["T1", arm, rep, str(study.ABLATION_SEED)]
        for s in study.MVSEC_SEQS:
            rel = study.m1_path(arm, study.ABLATION_SEED, study.stem("mvsec", s, dt, rep=rep))
            p = st.root / rel
            if not p.exists():
                line.append(notes.mark("expected, not present (results/eval/summary.json "
                                       "`missing`)"))
                continue
            doc = json.loads(p.read_text())
            if _check_tier(doc, rel) != "T1":
                raise TierError(f"{rel}: expected T1")
            st.sources.add(f"results/eval/{arm}/{study.ABLATION_SEED}/")
            line.append(fmt_cell({"metrics": doc["metrics"], "ci95": doc["ci95"],
                                  "seed_range": None, "seeds": [doc["seed"]]}, "epe"))
            ns[s].add(doc["n_windows"])
        body.append(line)
    head = ["tier", "arm", "rep.", "seed"] + [
        f"{s} (n={next(iter(ns[s]))})" if len(ns[s]) == 1 else s for s in study.MVSEC_SEQS]
    return _table(head, body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if results/TABLES.md is stale")
    ap.add_argument("--root", default=str(REPO), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.root)
    text = render(root)
    out = root / OUT_REL
    if args.check:
        old = out.read_text() if out.exists() else ""
        if old != text:
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(True), text.splitlines(True), OUT_REL, "regenerated"))
            print(f"{OUT_REL} is stale: run python scripts/make_tables.py", file=sys.stderr)
            return 1
        print(f"{OUT_REL} is current")
        return 0
    out.write_text(text)
    print(f"wrote {OUT_REL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
