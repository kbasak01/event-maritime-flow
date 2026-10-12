"""Phase 7 summary of the T3 evaluation matrix → ``results/eval_maritime/summary.json``.

Inventory of every expected unit (``evflow.eval.study_maritime``), then one entry per
(tier, arm, unit, slice, metric) for the tables:

- units: ``smd_test`` = the 3 smd_test videos pooled (mean of the per-video means; CI =
  stratified moving-block bootstrap, each video resampled independently, 1 s blocks),
  ``lars_eval_kf`` per slice (all / water / sky / obstacle) and ``lars_eval_all`` (cluster
  bootstrap over clips);
- 3-seed arms (real_sup, v2e_fixed_sup, v2e_rand_sup): per-window seed mean on the windows
  scored in every seed, its CI as above, and ``seed_range`` = min/max of the per-seed means;
- ``teacher`` block: teacher-mask coverage and event-pixel fraction per unit and slice (they do
  not depend on the model; read from the M0a units).

Per-window arrays come from ``runs/eval-maritime/npz`` and are checked against the SHA-256 each
unit JSON records. Nothing is chosen here.

    python scripts/summarise_maritime.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from check_data import sha256_file  # noqa: E402

from evflow.eval import maritime as em  # noqa: E402
from evflow.eval import study  # noqa: E402
from evflow.eval import study_maritime as sm  # noqa: E402
from evflow.eval.bootstrap import block_length, moving_block_ci, stratified_paired_ci  # noqa: E402
from evflow.sim.v2e_runner import git_dirty, git_sha  # noqa: E402

OUT = REPO / "results" / "eval_maritime" / "summary.json"
RES = dict(resamples=1000, ci=0.95, seed=0)
BLOCK_S = 1.0
FRAME_PX = 640 * 360


def _npz(doc: dict) -> dict[str, np.ndarray]:
    p = Path(doc["per_window_npz"].replace("~", str(Path.home()), 1))
    if not p.is_absolute():
        p = REPO / p
    if sha256_file(p) != doc["per_window_npz_sha256"]:
        raise SystemExit(f"{p}: SHA-256 differs from its unit JSON")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _combine(pws: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """(seed-mean series on windows finite in every seed, its validity mask, per-seed means)."""
    vals = np.stack([pw[key] for pw in pws])
    ok = np.isfinite(vals).all(0)
    per_seed = [float(v[ok].mean()) for v in vals] if ok.any() else []
    return vals.mean(0), ok, per_seed


def entry(tier, arm, model, seeds, unit, slc, metric, pws_by_seq) -> dict | None:
    """pws_by_seq: {sequence: [per-seed per-window dicts]}; >1 sequence = pooled SMD."""
    key = f"{metric}__{slc}"
    strata, blocks, per_seed_by_seq, n = [], [], [], 0
    cluster_vals = cluster_groups = None
    for _seq, pws in pws_by_seq.items():
        v, ok, ps = _combine(pws, key)
        if not ok.any():
            continue
        strata.append({"v": v[ok]})
        blocks.append(block_length(pws[0]["t_start_us"][ok], BLOCK_S))
        per_seed_by_seq.append(ps)
        n += int(ok.sum())
        cluster_vals, cluster_groups = v[ok], pws[0]["group"][ok]
    if not strata:
        return None
    if unit == "smd_test":
        mean, lo, hi = stratified_paired_ci(
            strata, lambda ss: np.mean([s["v"].mean(axis=-1) for s in ss], axis=0),
            blocks=blocks, **RES)
        per_seed = list(np.mean(np.array(per_seed_by_seq), axis=0))
        ci_method = "stratified moving-block bootstrap over windows (each video, 1 s blocks)"
    elif unit.startswith("smd_"):
        mean = float(strata[0]["v"].mean())
        lo, hi = moving_block_ci(strata[0]["v"], block=blocks[0], **RES)
        per_seed = per_seed_by_seq[0]
        ci_method = "moving-block bootstrap over windows (1 s blocks)"
    else:
        mean = float(cluster_vals.mean())
        lo, hi = em.cluster_ci(cluster_vals, cluster_groups, **RES)
        per_seed = per_seed_by_seq[0]
        ci_method = "cluster bootstrap over clips"
    return {"tier": tier, "arm": arm, "model": model, "seeds": seeds, "unit": unit,
            "slice": slc, "metric": metric, "mean": float(mean), "ci95": [float(lo), float(hi)],
            "seed_range": [float(min(per_seed)), float(max(per_seed))] if len(seeds) > 1
            else None, "n_windows": n, "ci_method": ci_method}


def main(argv: list[str] | None = None) -> int:
    units = sm.expected_units(REPO)
    present, missing = [], []
    for u in units:
        (present if (REPO / u["path"]).exists() else missing).append(u)
    docs = {u["path"]: json.loads((REPO / u["path"]).read_text()) for u in present}
    for u in present:
        d = docs[u["path"]]
        if d["tier"] != u["tier"] or d["arm"] != u["arm"]:
            raise SystemExit(f"{u['path']}: tier/arm {d['tier']}/{d['arm']} where "
                             f"{u['tier']}/{u['arm']} expected")
        if str(d["tier"]).startswith("T3") and "epe" in (d.get("metrics") or {}):
            raise SystemExit(f"{u['path']}: T3 record carries 'epe'")

    groups: dict[tuple, list[dict]] = {}
    for m in sm.models(REPO):
        groups.setdefault((m["arm"], m["model"]), []).append(m)
    entries = []
    smd = sm.smd_videos(REPO)
    for (arm, model), ms in groups.items():
        seeds = sorted(m["seed"] for m in ms if m["seed"] is not None) or [None]
        if arm in study.CORE_ARMS and len(seeds) != len(study.CORE_SEEDS):
            continue
        for tier, keys in (("T3a", em.T3A_KEYS), ("T3b", em.T3B_KEYS)):
            def pws(ds, seq, _ms=ms, _tier=tier):
                out = []
                for m in _ms:
                    p = sm.t3_path(m["arm"], m["seed"], ds, seq, _tier)
                    if p not in docs:
                        return None
                    out.append(_npz(docs[p]))
                return out
            by_unit = {"smd_test": {v: pws("smd", v) for v in smd}}
            for v in smd:
                by_unit[f"smd_{v}"] = {v: by_unit["smd_test"][v]}
            by_unit["lars_eval_kf"] = {"kf": pws("lars", "lars_eval_kf")}
            by_unit["lars_eval_all"] = {"all": pws("lars", "lars_eval_all")}
            for unit, pbs in by_unit.items():
                if any(p is None for p in pbs.values()):
                    continue
                slices = em.SLICES if unit == "lars_eval_kf" else ("all",)
                for slc in slices:
                    for k in keys:
                        e = entry(tier, arm, model, seeds, unit, slc, k, pbs)
                        if e is not None:
                            entries.append(e)

    teacher = {}
    for unit, ds, seqs in (("smd_test", "smd", smd), ("lars_eval_kf", "lars", ["lars_eval_kf"]),
                           ("lars_eval_all", "lars", ["lars_eval_all"])):
        pw_list = []
        for s in seqs:
            p = sm.t3_path("zero_flow", None, ds, s, "T3a")
            if p in docs:
                pw_list.append(_npz(docs[p]))
        if len(pw_list) != len(seqs):
            continue
        teacher[unit] = {}
        for slc in (em.SLICES if unit == "lars_eval_kf" else ("all",)):
            row = {}
            for k in ("teacher_mask_coverage", "events_px_fraction"):
                v = np.concatenate([pw[f"{k}__{slc}"] for pw in pw_list])
                v = v[np.isfinite(v)]
                row[k] = float(v.mean()) if v.size else None
            sp = np.concatenate([pw[f"slice_px__{slc}"] for pw in pw_list])
            row["mean_slice_px_fraction"] = float(np.nanmean(sp) / FRAME_PX)
            row["n_windows"] = int(np.isfinite(sp).sum())
            teacher[unit][slc] = row

    doc = {
        "kind": "maritime_eval_summary", "phase": 7,
        "tier_note": "T3a (RAFT teacher-EPE family) and T3b (FWL/RSAT) entries carry their "
                     "tier; the T1 rows of the maritime arms on mvsec_test are the unit JSONs in "
                     "results/eval/<arm>/1337/",
        "n_expected": len(units), "n_present": len(present),
        "missing": [{"path": u["path"], "tier": u["tier"]} for u in missing],
        "units_by_tier": {t: sum(u["tier"] == t for u in present) for t in ("T1", "T3a",
                                                                               "T3b")},
        "bootstrap": {**RES, "block_seconds": BLOCK_S},
        "protocol": {"frame_rate": "SMD 30 fps (ffprobe); LaRS assumed 30 fps (D-024); flow in "
                                   "px per frame interval at 640x360 (LaRS stretched, D-054)",
                     "warmup_windows": em.WARMUP},
        "teacher": teacher,
        "entries": entries,
        "git_sha": git_sha(), "git_dirty": git_dirty(),
        "created_by": "scripts/summarise_maritime.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(present)}/{len(units)} units present, {len(entries)} entries -> {OUT}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
