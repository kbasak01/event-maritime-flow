"""Summarise the v2e throughput sweep into results/v2e_throughput.json (Phase 2 step 2).

    python scripts/summarise_throughput.py --units data/converted/v2e/throughput \
        --logs runs/v2e-throughput [--budget configs/v2e_budget.yaml]

Per unit: video seconds, wall time, v2e stage times parsed from the v2e log (SuperSloMo and DVS
emulation progress bars), fixed overhead (= wall − stages), compute seconds per video second
(wall and marginal), events/s, upsampling factor, warnings. Then every corpus scenario in the
budget config is costed as ``n_units × overhead + video_s × marginal_rate`` (LaRS: per clip), so
the corpus cuts in docs/DECISIONS.md cite numbers produced here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
BAR = re.compile(r"^(?P<name>slomo-interp|dvs): +100%\|[^|]*\| *\d+/\d+ \[(?P<t>[\d:]+)<")
UPS = re.compile(r"SloMo average upsampling factor= *([\d.]+)")


def _secs(s: str) -> int:
    parts = [int(p) for p in s.split(":")]
    out = 0
    for p in parts:
        out = out * 60 + p
    return out


def stage_times(log_text: str) -> dict:
    """Elapsed seconds of the final 100 % progress bar of each v2e stage."""
    out: dict = {}
    for line in log_text.replace("\r", "\n").splitlines():
        m = BAR.match(line.strip())
        if m:
            out[m.group("name")] = _secs(m.group("t"))
    ups = UPS.findall(log_text)
    return {"slomo_s": out.get("slomo-interp"), "dvs_s": out.get("dvs"),
            "upsampling_factor": float(ups[-1]) if ups else None}


def video_seconds(m: dict) -> float:
    inp = m["input"]
    return (int(inp["n_frames"]) - 1) / float(m["input_frame_rate"])


def unit_row(m: dict, log_text: str) -> dict:
    vs = video_seconds(m)
    st = stage_times(log_text)
    wall = float(m["wall_time_s"])
    stages = (st["slomo_s"] or 0) + (st["dvs_s"] or 0)
    conv = m["conversion"]
    return {
        "unit": m["unit"], "dataset": m["input"]["dataset"], "profile": m["profile"],
        "overrides": m["overrides"], "video_s": round(vs, 4), "wall_s": wall,
        "slomo_s": st["slomo_s"], "dvs_s": st["dvs_s"], "overhead_s": round(wall - stages, 1),
        "upsampling_factor": st["upsampling_factor"],
        "wall_per_video_s": round(wall / vs, 2), "marginal_per_video_s": round(stages / vs, 2),
        "n_events": conv["n_events"], "events_per_video_s": round(conv["n_events"] / vs),
        "on_fraction": conv["on_fraction"], "undersampling_warnings": len(
            m["warnings"]["undersampling"]),
        "gpu": m["gpu"], "gpu_mem_used_peak_mib": m["gpu_mem_used_peak_mib"],
        "store_bytes": m["store_bytes"],
    }


def measured_item(item: dict) -> dict:
    """A budget line already run: GPU hours = sum of its units' wall times / their concurrency
    (units overlapped, so each unit's wall already includes the contention), read from the
    committed results JSON that lists them (results/v2e_calibration.json: cells[].units[];
    results/v2e_mvsec_corpus.json: units[], optionally only the corpora in ``corpora``)."""
    doc = json.loads((REPO / item["measured_from"]).read_text())
    if "cells" in doc:      # optionally only one grid's cells (the timing JSON lists both)
        walls = [u["wall_time_s"] for c in doc["cells"]
                 if item.get("grid") is None or c.get("grid") == item["grid"] for u in c["units"]]
    else:
        keep = item.get("corpora")
        walls = [u["wall_time_s"] for u in doc["units"] if keep is None or u["corpus"] in keep]
    n = int(item["concurrency"])
    return {**item, "units": len(walls), "sum_unit_wall_s": round(sum(walls), 1),
            "gpu_hours": round(sum(walls) / n / 3600, 3), "measured": True}


def sweep_item(item: dict, rows: dict[str, dict], conc: list[dict]) -> dict:
    """The throughput sweep itself (already spent): sequential units at their wall time,
    concurrent batches at their batch wall time, plus the trial run's manifest if present."""
    par = {u for c in conc for u in c["units"]}
    seq = sum(r["wall_s"] for k, r in rows.items() if k not in par)
    batches = sum(c["batch_wall_s"] for c in conc)
    trial = REPO / item.get("trial_manifest", "")
    trial_s = json.loads(trial.read_text())["wall_time_s"] if trial.is_file() else 0.0
    return {**item, "sequential_units_wall_s": round(seq, 1),
            "concurrent_batches_wall_s": round(batches, 1), "trial_wall_s": trial_s,
            "gpu_hours": round((seq + batches + trial_s) / 3600, 3), "measured": True}


def cost_item(item: dict, rows: dict[str, dict], conc: list[dict] | None = None) -> dict:
    """GPU-seconds of one budget line, from the rate of a measured unit (or group of units)."""
    if "measured_from" in item:
        return measured_item(item)
    if item.get("measured_sweep"):
        return sweep_item(item, rows, conc or [])
    ref = item["rate_from"]
    refs = [r for k, r in rows.items() if re.fullmatch(ref, k)]
    if not refs:
        raise KeyError(f"no measured unit matches {ref!r}")
    overhead = float(np.median([r["overhead_s"] for r in refs]))
    if "clips" in item:
        per_clip = float(np.median([r["wall_s"] for r in refs]))
        gpu_s = item["clips"] * per_clip * item.get("multiplier", 1)
        basis = {"per_clip_wall_s": per_clip}
    else:
        rate = float(np.median([r["marginal_per_video_s"] for r in refs]))
        n_units = item.get("units") or max(1, int(np.ceil(item["video_s"] / item.get(
            "unit_len_s", 60))))
        gpu_s = (item["video_s"] * rate + n_units * overhead) * item.get("multiplier", 1)
        basis = {"marginal_per_video_s": rate, "overhead_s": overhead, "units": n_units}
    return {**item, **basis, "gpu_hours": round(gpu_s / 3600, 3)}


def concurrency_factor(conc: list[dict], rows: dict, n: int, reference: str) -> float:
    """Wall time of n overlapping units relative to n sequential runs of the reference unit
    (1.0 for n = 1). Measured on MVSEC; applied to every item of a scenario (assumption)."""
    if n == 1:
        return 1.0
    c = next((c for c in conc if c["concurrency"] == n), None)
    if c is None or reference not in rows:
        raise KeyError(f"no concurrency-{n} measurement or reference unit {reference!r}")
    single = rows[reference]["wall_per_video_s"]
    return round(c["effective_gpu_s_per_video_s"] / single, 3)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--budget", type=Path, default=REPO / "configs/v2e_budget.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "results/v2e_throughput.json")
    a = ap.parse_args(argv)

    rows = {}
    for mf in sorted(a.units.glob("*.json")):
        if mf.name.endswith((".v2e.json", ".batch.json")):
            continue
        m = json.loads(mf.read_text())
        rows[m["unit"]] = unit_row(m, (a.units / f"{m['unit']}.log").read_text(errors="replace"))
    failed = sorted(p.stem for p in a.logs.glob("*.log")
                    if p.stem not in rows and not p.stem.startswith("job"))
    doc = {
        "kind": "v2e_throughput",
        "tier": "n/a",
        "tier_note": "compute cost of the simulator, not an evidence-tier result",
        "plan_step": "Phase 2 step 2",
        "note": "compute cost of v2e on this workstation; not an evidence-tier metric",
        "units": list(rows.values()),
        "failed_units": failed,
        "lars_clips_summary": None,
    }
    conc = []
    for bf in sorted(a.units.glob("concurrency_*.batch.json")):
        b = json.loads(bf.read_text())
        n = int(b["concurrency"])
        members = [r for k, r in rows.items() if re.fullmatch(rf".*_par{n}_\d+", k)]
        if len(members) != n:
            continue
        vs = sum(r["video_s"] for r in members)
        conc.append({"concurrency": n, "units": [r["unit"] for r in members],
                     "batch_wall_s": round(float(b["batch_wall_s"]), 1),
                     "effective_gpu_s_per_video_s": round(float(b["batch_wall_s"]) / vs, 2)})
    doc["concurrency"] = conc
    lars = [r for r in rows.values() if r["dataset"] == "lars"]
    if lars:
        doc["lars_clips_summary"] = {
            "n": len(lars), "wall_s_median": float(np.median([r["wall_s"] for r in lars])),
            "wall_s_p90": float(np.percentile([r["wall_s"] for r in lars], 90)),
            "video_s_median": float(np.median([r["video_s"] for r in lars]))}
    if a.budget.exists():
        bud = yaml.safe_load(a.budget.read_text())
        doc["budget_config"] = (str(a.budget.resolve().relative_to(REPO))
                                if a.budget.resolve().is_relative_to(REPO) else str(a.budget))
        doc["budget_limit_gpu_hours"] = bud["limit_gpu_hours"]
        doc["scenarios"] = {}
        for name, sc in bud["scenarios"].items():
            items = [cost_item(it, rows, conc) for it in sc["items"]]
            done = sum(i["gpu_hours"] for i in items if i.get("measured"))
            seq = sum(i["gpu_hours"] for i in items if not i.get("measured"))
            n = int(sc.get("concurrency", 1))
            eff = concurrency_factor(conc, rows, n, bud.get("concurrency_reference", ""))
            total = done + seq * eff
            doc["scenarios"][name] = {
                "description": sc.get("description", ""), "items": items,
                "measured_gpu_hours": round(done, 2),
                "sequential_gpu_hours": round(seq, 2), "concurrency": n,
                "concurrency_factor": eff, "total_gpu_hours": round(total, 2),
                "within_limit": bool(total <= bud["limit_gpu_hours"])}
    doc.update({"git_sha": git_sha(), "created_by": "scripts/summarise_throughput.py",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    for r in rows.values():
        print(f"{r['unit']:28s} video {r['video_s']:6.2f}s wall {r['wall_s']:7.1f}s "
              f"x{r['wall_per_video_s']:6.1f} marginal x{r['marginal_per_video_s']:6.1f} "
              f"ups {r['upsampling_factor']} ev/s {r['events_per_video_s']:.3g} "
              f"warn {r['undersampling_warnings']}")
    for n, sc in doc.get("scenarios", {}).items():
        print(f"scenario {n}: {sc['total_gpu_hours']} GPU-h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
