"""Phase 6 summary of every evaluation unit → three JSON files read by ``scripts/make_tables.py``.

    python scripts/summarise_eval.py

* ``results/eval/summary.json`` (``kind: eval_inventory``): the expected unit matrix of
  ``evflow.eval.study`` against the JSON present; missing units; git provenance; smoke or dirty
  units (both must be none); checkpoint SHA against each run's ``selection.json`` (when the run
  directory is present); the reproduction check (``runs/eval/repro/``: one unit re-evaluated at
  the job's commit must equal its committed JSON to 1e-6).
* ``results/eval/aggregate.json`` (``kind: eval_seed_aggregate``): per (tier, arm,
  representation, unit) the M1 seed mean, the per-seed values and their range, and the moving-block
  bootstrap CI of the seed-mean per-window series (windows scored in every seed).
* ``results/eval/paired.json`` (``kind: eval_paired_bootstrap``): EPE gaps with paired moving-block
  bootstrap CIs on the same windows — Table B (gap = v2e arm − real arm, 3-seed means; closure =
  1 − gap(v2e_rand_sup) / gap(v2e_fixed_sup), undefined when the CI of gap(v2e_fixed_sup) reaches
  0; self-supervised gap = v2e_selfsup − real_selfsup), Table C (v2e_fixed_sup − real_sup per
  representation, seed 1337), and T2 − T1 per model (the same model on v2e vs real test events).
  Pooled rows average the four sequence means and resample each sequence independently.

Inputs are the per-unit JSON in ``results/`` and their per-window npz (SHA-256 verified; the npz
are local, gitignored). Bootstrap settings: ``configs/evflow.yaml metrics.bootstrap``, seed 0.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from check_data import sha256_file

from evflow.eval import study
from evflow.eval.bootstrap import (
    block_length,
    moving_block_ci,
    paired_block_ci,
    stratified_paired_ci,
)
from evflow.sim.v2e_runner import git_dirty, git_sha

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "eval"
REPRO = REPO / "runs" / "eval" / "repro"
METRICS = ("epe", "outlier_pct", "npe_1", "npe_2", "npe_3", "ae_deg", "fwl")
REPRO_TOL = 1e-6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _boot() -> dict:
    b = yaml.safe_load((REPO / "configs/evflow.yaml").read_text())["metrics"]["bootstrap"]
    return {"block_seconds": float(b["block_seconds"]), "resamples": int(b["resamples"]),
            "ci": float(b["ci"]), "seed": 0}


def _load_npz(doc: dict, cache: dict) -> dict[str, np.ndarray]:
    path = Path(doc["per_window_npz"]).expanduser()
    if path not in cache:
        if sha256_file(path) != doc["per_window_npz_sha256"]:
            raise SystemExit(f"{path}: SHA-256 differs from its summary JSON")
        with np.load(path) as z:
            cache[path] = {k: z[k] for k in z.files}
    return cache[path]


def _scored(pw: dict, metric: str) -> np.ndarray:
    ok = np.isfinite(pw[metric])
    return ok if metric == "fwl" else ok & (pw["n_px"] > 0)


class Units:
    """Per-unit JSON and per-window arrays, keyed by (tier, arm, rep, dataset, seq, dt)."""

    def __init__(self, present: list[dict]) -> None:
        self.groups: dict[tuple, list[dict]] = defaultdict(list)
        self.npz: dict = {}
        for u in present:
            rep = u.get("representation") if u["kind"] == "m1" else "baseline"
            self.groups[(u["tier"], u["arm"], rep, u["dataset"], u["sequence"],
                         u["dt"])].append(u)

    def per_window(self, key: tuple, seeds: tuple[int, ...] | None = None) -> list[dict]:
        units = self.groups.get(key, [])
        if seeds is not None:
            units = [u for u in units if u.get("seed") in seeds]
        out = []
        for u in sorted(units, key=lambda u: u.get("seed") or 0):
            out.append(_load_npz(u["doc"], self.npz))
        if out:
            ref = out[0]["index"]
            if any(not np.array_equal(p["index"], ref) for p in out[1:]):
                raise SystemExit(f"{key}: seeds evaluated on different windows")
        return out

    def series(self, key: tuple, metric: str = "epe",
               seeds: tuple[int, ...] | None = None) -> dict | None:
        """Seed-mean per-window series (all windows; scored = scored in every seed)."""
        pws = self.per_window(key, seeds)
        if not pws:
            return None
        ok = np.logical_and.reduce([_scored(p, metric) for p in pws])
        vals = np.mean([np.where(ok, p[metric], 0.0) for p in pws], axis=0)
        return {"index": pws[0]["index"], "t_start_us": pws[0]["t_start_us"], "ok": ok,
                "v": vals, "n_seeds": len(pws)}


def inventory(expected: list[dict], runs: list[dict]) -> tuple[list[dict], dict]:
    present, missing, problems = [], [], []
    shas, dirty, smoke = set(), [], []
    sel_sha = {}
    runs_root = Path(yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
                     ["outputs"]["runs"]) / "train"
    for r in runs:
        sel = runs_root / r["run_id"] / "selection.json"
        if sel.exists():
            sel_sha[(r["arm"], int(r["seed"]), r["representation"])] = json.loads(
                sel.read_text())["checkpoint_sha256"]
    ck_checked = 0
    for u in expected:
        p = REPO / u["path"]
        if not p.exists():
            missing.append({k: u[k] for k in ("path", "tier", "arm", "dataset", "sequence", "dt")})
            continue
        doc = json.loads(p.read_text())
        for k, v in (("tier", u["tier"]), ("arm", u["arm"]), ("dataset", u["dataset"]),
                     ("sequence", u["sequence"])):
            if doc.get(k) != v:
                problems.append(f"{u['path']}: {k}={doc.get(k)!r}, expected {v!r}")
        if u["kind"] == "m1":
            if doc.get("representation") != u["representation"] or doc.get("seed") != u["seed"]:
                problems.append(f"{u['path']}: representation/seed mismatch")
            key = (u["arm"], u["seed"], u["representation"])
            if key in sel_sha:
                ck_checked += 1
                if sel_sha[key] != doc.get("checkpoint_sha256"):
                    problems.append(f"{u['path']}: checkpoint SHA differs from selection.json")
        shas.add(doc.get("git_sha"))
        if doc.get("git_dirty"):
            dirty.append(u["path"])
        if doc.get("smoke"):
            smoke.append(u["path"])
        present.append({**u, "doc": doc})
    inv = {"n_expected": len(expected), "n_present": len(present), "n_missing": len(missing),
           "missing": missing, "by_design": study.BY_DESIGN, "problems": problems,
           "git_shas": sorted(s for s in shas if s), "git_dirty_units": dirty,
           "smoke_units": smoke, "checkpoint_sha_checked_against_selection": ck_checked}
    return present, inv


def repro_check() -> dict:
    out = {"rule": f"metrics and ci95 of the re-evaluated unit equal the committed JSON to "
                   f"{REPRO_TOL:g}", "units": []}
    for f in sorted(REPRO.rglob("*.json")):
        rel = f.relative_to(REPRO)
        ref = REPO / "results" / "eval" / rel
        if not ref.exists():
            continue
        a, b = json.loads(f.read_text()), json.loads(ref.read_text())
        diffs = [abs(a["metrics"][k] - b["metrics"][k]) for k in a["metrics"]
                 if a["metrics"][k] is not None and b["metrics"][k] is not None]
        diffs += [abs(x - y) for k in a["ci95"] if a["ci95"][k] and b["ci95"][k]
                  for x, y in zip(a["ci95"][k], b["ci95"][k], strict=True)]
        md = max(diffs) if diffs else None
        out["units"].append({"unit": f"results/eval/{rel}", "committed_git_sha": b["git_sha"],
                             "repro_git_sha": a["git_sha"], "max_abs_diff": md,
                             "n_windows_equal": a["n_windows"] == b["n_windows"],
                             "pass": md is not None and md <= REPRO_TOL
                             and a["n_windows"] == b["n_windows"]})
    out["pass"] = bool(out["units"]) and all(u["pass"] for u in out["units"])
    return out


def aggregates(units: Units, boot: dict) -> list[dict]:
    entries = []
    for key, us in sorted(units.groups.items(), key=lambda kv: str(kv[0])):
        tier, arm, rep, ds, seq, dt = key
        if rep == "baseline":
            continue
        us = sorted(us, key=lambda u: u["seed"])
        seeds = [u["seed"] for u in us]
        e = {"tier": tier, "arm": arm, "model": "M1", "representation": rep, "dataset": ds,
             "sequence": seq, **({"dt": dt} if ds == "mvsec" else {"window_ms": 100}),
             "events": us[0]["events"], "seeds": seeds,
             "seeds_complete": sorted(seeds) == sorted(study.expected_seeds(arm, rep)),
             "checkpoint_sha256": {str(u["seed"]): u["doc"]["checkpoint_sha256"] for u in us},
             "source_json": [u["path"] for u in us], "metrics": {}, "per_seed": {},
             "seed_range": {}, "ci95": {}, "n_windows": {}}
        for m in METRICS:
            per_seed = {str(u["seed"]): u["doc"]["metrics"][m] for u in us}
            e["per_seed"][m] = per_seed
            s = units.series(key, m)
            if s is None or not s["ok"].any():
                e["metrics"][m] = e["ci95"][m] = e["seed_range"][m] = None
                continue
            v = s["v"][s["ok"]]
            e["metrics"][m] = float(v.mean())
            vals = [x for x in per_seed.values() if x is not None]
            e["seed_range"][m] = [min(vals), max(vals)] if vals else None
            # the per-seed JSON means are over the same windows, so their mean is the seed mean
            if vals and abs(np.mean(vals) - e["metrics"][m]) > 1e-9:
                raise SystemExit(f"{key} {m}: seed mean {e['metrics'][m]} != mean of JSON "
                                 f"{np.mean(vals)} (scored windows differ between seeds)")
            block = block_length(s["t_start_us"], boot["block_seconds"])
            e["ci95"][m] = list(moving_block_ci(v, block=block, resamples=boot["resamples"],
                                                ci=boot["ci"], seed=boot["seed"]))
            e["n_windows"][m] = int(s["ok"].sum())
            e["block_windows"] = block
        entries.append(e)
    return entries


def _mean(a):
    return a.mean(-1)


def _strata(units: Units, spec: dict[str, tuple], seqs: list[str], dt) -> list[dict] | None:
    """Per sequence: the named seed-mean EPE series on the windows scored in all of them."""
    out = []
    for seq in seqs:
        ss = {}
        for name, (tier, arm, rep, ds, seeds) in spec.items():
            s = units.series((tier, arm, rep, ds, seq, dt), "epe", seeds)
            want = len(seeds) if seeds else (len(study.expected_seeds(arm, rep))
                                             if rep != "baseline" else 1)
            if s is None or s["n_seeds"] != want:
                return None     # incomplete: rendered as a missing cell
            ss[name] = s
        idx = [s["index"] for s in ss.values()]
        if any(not np.array_equal(i, idx[0]) for i in idx[1:]):
            raise SystemExit(f"{seq} dt{dt}: series on different windows")
        ok = np.logical_and.reduce([s["ok"] for s in ss.values()])
        first = next(iter(ss.values()))
        out.append({"sequence": seq, "t": first["t_start_us"][ok],
                    "series": {k: s["v"][ok] for k, s in ss.items()}})
    return out


def _gap(a: str, b: str):
    return lambda s: _mean(s[a]) - _mean(s[b])


def _closure(s):
    return 1.0 - (_mean(s["rand"]) - _mean(s["real"])) / (_mean(s["fixed"]) - _mean(s["real"]))


def _pooled(fn):
    """Statistic of per-sequence means averaged over sequences (closure: ratio of pooled gaps)."""
    def stat(strata):
        means = [{k: _mean(v)[..., None] for k, v in st.items()} for st in strata]
        avg = {k: np.mean([m[k] for m in means], axis=0) for k in means[0]}
        return fn(avg)
    return stat


def paired_entry(units: Units, boot: dict, *, table: str, tier: str, comparison: str,
                 spec: dict, stat, seqs: list[str], dt, ds: str, definition: str,
                 pooled: bool, same_seed: bool = False) -> dict | None:
    strata = _strata(units, spec, seqs, dt)
    if strata is None:
        return None
    blocks = [block_length(st["t"], boot["block_seconds"]) for st in strata]
    kw = {"resamples": boot["resamples"], "ci": boot["ci"], "seed": boot["seed"]}
    if pooled:
        est, lo, hi = stratified_paired_ci([st["series"] for st in strata], _pooled(stat),
                                           blocks=blocks, **kw)
    else:
        est, lo, hi = paired_block_ci(strata[0]["series"], stat, block=blocks[0], **kw)
    series = {}
    for k, (t, arm, rep, d, seeds) in spec.items():
        us = [u for seq in seqs for u in units.groups.get((t, arm, rep, d, seq, dt), [])
              if seeds is None or u.get("seed") in seeds]
        series[k] = {"tier": t, "arm": arm, "representation": rep,
                     "seeds": list(seeds) if seeds else "all",
                     "source_json": sorted(u["path"] for u in us),
                     "checkpoint_sha256": sorted({u["doc"]["checkpoint_sha256"] for u in us
                                                  if u["doc"].get("checkpoint_sha256")})}
    out = {"table": table, "tier": tier, "comparison": comparison, "metric": "epe",
           "dataset": ds, "sequence": "pooled" if pooled else seqs[0],
           **({"dt": dt} if ds == "mvsec" else {}),
           "estimate": est, "ci95": [lo, hi],
           "ci_scope": "windows only (each arm's seed mean held fixed); between-seed variation "
                       "is in seed_pairings",
           "n_windows": {st["sequence"]: int(st["t"].size) for st in strata},
           "block_windows": dict(zip([st["sequence"] for st in strata], blocks, strict=True)),
           "series": series, "definition": definition}
    sp = seed_pairings(units, spec, stat, seqs, dt, pooled, same_seed=same_seed)
    if sp is not None:
        out["seed_pairings"] = sp
    return out


def seed_pairings(units: Units, spec: dict, stat, seqs: list[str], dt, pooled: bool, *,
                  same_seed: bool) -> dict | None:
    """Point estimate of the statistic per combination of single training seeds of the
    3-seed series (D-053): every pairing across arms (independent seeds, 3^k combinations),
    or the same seed in every series (``same_seed``: one model on two event streams)."""
    multi = [k for k, (_, arm, rep, _, seeds) in spec.items()
             if seeds is None and arm in study.CORE_ARMS and rep == study.MAIN_REP]
    if not multi:
        return None
    combos = ([dict.fromkeys(multi, s) for s in study.CORE_SEEDS] if same_seed else
              [dict(zip(multi, c, strict=True)) for c in
               itertools.product(study.CORE_SEEDS, repeat=len(multi))])
    values = {}
    for c in combos:
        sp = {k: (*v[:4], (c[k],) if k in c else v[4]) for k, v in spec.items()}
        strata = _strata(units, sp, seqs, dt)
        if strata is None:
            return None
        ser = [st["series"] for st in strata]
        v = _pooled(stat)(ser) if pooled else stat(ser[0])
        values[",".join(f"{k}:{c[k]}" for k in multi)] = float(np.asarray(v).item())
    vals = list(values.values())
    return {"rule": "same seed in every series" if same_seed else
            "all combinations of single seeds across arms", "n": len(vals),
            "range": [min(vals), max(vals)], "values": values}


def mark_seed_stability(closure: dict, gap: dict | None) -> None:
    """D-053: closure is seed-unstable when gap(v2e_fixed_sup) <= 0 for any single-seed pairing
    (fixed seed, real seed); its pairing range then covers only pairings with a positive
    denominator, and the count of the others is recorded."""
    sp, gsp = closure.get("seed_pairings"), (gap or {}).get("seed_pairings")
    if sp is None or gsp is None:
        return
    def den(key):
        d = dict(kv.split(":") for kv in key.split(","))
        return gsp["values"][f"v2e:{d['fixed']},real:{d['real']}"]
    ok = {k: v for k, v in sp["values"].items() if den(k) > 0}
    bad_gaps = sum(v <= 0 for v in gsp["values"].values())
    closure["seed_unstable"] = bad_gaps > 0
    closure["n_fixed_gap_pairings_nonpositive"] = f"{bad_gaps} of {gsp['n']}"
    sp["range"] = [min(ok.values()), max(ok.values())] if ok else None
    sp["n_defined"] = len(ok)


def paired(units: Units, boot: dict) -> list[dict]:
    entries = []
    rep = study.MAIN_REP
    core = None   # all seeds present

    def add(**kw):
        for pooled, seqs in [(False, [s]) for s in study.MVSEC_SEQS] + [(True,
                                                                        list(study.MVSEC_SEQS))]:
            e = paired_entry(units, boot, seqs=seqs, pooled=pooled, ds="mvsec", **kw)
            if e is not None:
                entries.append(e)
                if kw["comparison"] == "gap_v2e_fixed_sup":
                    gaps[(kw["tier"], kw["dt"], e["sequence"])] = e

    gaps: dict = {}
    for tier in ("T1", "T2"):
        for dt in study.DTS:
            for arm in ("v2e_fixed_sup", "v2e_rand_sup"):
                add(table="B", tier=tier, comparison=f"gap_{arm}", dt=dt,
                    spec={"v2e": (tier, arm, rep, "mvsec", core),
                          "real": (tier, "real_sup", rep, "mvsec", core)},
                    stat=_gap("v2e", "real"),
                    definition=f"EPE({arm}) - EPE(real_sup), 3-seed means per window")
            before = len(entries)
            add(table="B", tier=tier, comparison="closure_v2e_rand_sup", dt=dt,
                spec={"fixed": (tier, "v2e_fixed_sup", rep, "mvsec", core),
                      "rand": (tier, "v2e_rand_sup", rep, "mvsec", core),
                      "real": (tier, "real_sup", rep, "mvsec", core)},
                stat=_closure,
                definition="1 - gap(v2e_rand_sup) / gap(v2e_fixed_sup), 3-seed means; undefined "
                           "(null) when the CI of gap(v2e_fixed_sup) includes 0 or lies below")
            for e in entries[before:]:
                g = gaps.get((tier, dt, e["sequence"]))
                if g is None or g["ci95"][0] <= 0:
                    e.update(estimate=None, ci95=None, undefined_reason=(
                        "gap(v2e_fixed_sup) CI includes 0 or lies below it"))
                mark_seed_stability(e, g)
            add(table="B", tier=tier, comparison="gap_selfsup", dt=dt,
                spec={"v2e": (tier, "v2e_selfsup", rep, "mvsec", core),
                      "real": (tier, "real_selfsup", rep, "mvsec", core)},
                stat=_gap("v2e", "real"),
                definition="EPE(v2e_selfsup) - EPE(real_selfsup), one seed each")
    for r in study.ABLATION_REPS:
        for dt in study.DTS:
            s1 = (study.ABLATION_SEED,)
            add(table="C", tier="T1", comparison=f"gap_v2e_fixed_sup_{r}", dt=dt,
                spec={"v2e": ("T1", "v2e_fixed_sup", r, "mvsec", s1),
                      "real": ("T1", "real_sup", r, "mvsec", s1)},
                stat=_gap("v2e", "real"),
                definition=f"EPE(v2e_fixed_sup) - EPE(real_sup), {r}, seed {study.ABLATION_SEED}")
    for arm in study.M1_ARMS + ("zero_flow",):
        r = "baseline" if arm == "zero_flow" else rep
        for dt in study.DTS:
            add(table="T1_vs_T2", tier="T2", comparison=f"t2_minus_t1_{arm}", dt=dt,
                spec={"t2": ("T2", arm, r, "mvsec", core), "t1": ("T1", arm, r, "mvsec", core)},
                stat=_gap("t2", "t1"), same_seed=True,
                definition=f"EPE({arm} on v2e test events) - EPE(same model on real test events),"
                           " same windows and GT; each stream's own sparse mask")
    return entries


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {path.relative_to(REPO)}")


def main() -> int:
    runs = study.trained_runs(REPO)
    expected = study.expected_units(runs)
    present, inv = inventory(expected, runs)
    boot = _boot()
    units = Units(present)
    common = {"git_sha": git_sha(), "git_dirty": git_dirty(),
              "created_by": "scripts/summarise_eval.py", "created_at": _now()}
    inv["repro_check"] = repro_check()
    _write(OUT / "summary.json", {
        "kind": "eval_inventory", "tier": "n/a",
        "tier_note": "inventory of the Phase 6 units; the units carry their own tiers",
        "phase": 6, "bootstrap": boot, **inv, **common})
    _write(OUT / "aggregate.json", {
        "kind": "eval_seed_aggregate", "tier_note": "each entry carries its tier",
        "bootstrap": boot, "entries": aggregates(units, boot), **common})
    _write(OUT / "paired.json", {
        "kind": "eval_paired_bootstrap", "tier_note": "each entry carries its tier",
        "bootstrap": {**boot, "method": "paired moving-block (same window indices for every "
                      "series); pooled rows: independent per sequence, mean of sequence means"},
        "entries": paired(units, boot), **common})
    print(f"units {inv['n_present']}/{inv['n_expected']}, missing {inv['n_missing']}, "
          f"problems {len(inv['problems'])}, repro {inv['repro_check']['pass']}")
    return 0 if not inv["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
