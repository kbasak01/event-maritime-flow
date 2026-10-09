"""Simulator-fidelity report: §1.6 event statistics, real DVS vs v2e(APS) (Phase 3 step 2).

    python scripts/fidelity_report.py [--group test_indoor_flying1 ...] [--no-figures]

Groups (units of the corpus plan, ``sim/corpus.py``, profile ``mvsec_cal``): ``tune_outdoor_day2``
(the whole tune slice), ``test_<sequence>`` for each ``mvsec_test`` sequence, ``train_cal`` (the 4
train clips). Statistics, windows and spans follow ``sim/fidelity.py`` (D-033), identically for
real and v2e. Writes ``results/fidelity/<group>.json`` (histograms on fixed edges, so every W1 is
recomputable), ``results/fidelity/summary.json`` and figures in ``results/fidelity/figures/``
(gitignored: MVSEC-derived imagery, D-008). Tier T2 statistics; **reported only, never used for
any choice** (plan Phase 3 step 2) — nothing in the repo reads these files to select anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from evflow.io.mvsec import read_aps_timestamps_us
from evflow.io.store import EventSlicer, EventStore
from evflow.sim import corpus as K
from evflow.sim import fidelity as F
from evflow.sim import stats as S
from evflow.sim.v2e_runner import redact_home

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "fidelity"
WIDTH, HEIGHT = 346, 260
REAL_C, V2E_C = "#2a78d6", "#eb6834"          # dataviz palette slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NOTES = [
    "Reported only: test-sequence statistics are never used for any choice (plan Phase 3 step 2).",
    "iei: v2e distributes the events of one pixel within one (interpolated) frame interval in a "
    "fixed timestamp 'pyramid' (v2e README); short inter-event intervals of v2e reflect this "
    "artefact and the 0.75 ms timestamp cap (D-025), not sensor physics (D-027).",
    "hot_fraction: one sample per 500 ms block, so its histogram rests on n_blocks samples per "
    "group (see n_samples.hot_fraction); treat its W1 as coarse.",
    "Both streams use identical windows, blocks and spans; real DVS hot pixels and noise are "
    "kept (no filtering of either stream).",
    "v2e inherits the APS frames' dynamic range, blur and frame rate; nothing here measures HDR "
    "or latency behaviour.",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def groups(plan: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for u in plan["units"]:
        if u["corpus"] == "tune":
            out.setdefault(f"tune_{u['sequence']}", []).append(u)
        elif u["corpus"] == "test":
            out.setdefault(f"test_{u['sequence']}", []).append(u)
        elif u["corpus"] == "train_cal":
            out.setdefault("train_cal", []).append(u)
    return out


def read_syn(path: Path) -> dict:
    with EventStore(path) as s:
        return {"t": s.events["t"][:] + s.t_offset, "x": s.events["x"][:],
                "y": s.events["y"][:], "p": s.events["p"][:]}


def read_real(store: Path, t0: int, t1: int) -> dict:
    with EventSlicer(store) as sl:
        e = sl.get_events(t0, t1)
    if e is None:
        raise RuntimeError(f"{store} does not cover [{t0}, {t1})")
    return e


def rate_image(e: dict, span: list[int]) -> np.ndarray:
    r = F.restrict(e, *span)
    return S.count_image(r, WIDTH, HEIGHT)


def run_group(name: str, units: list[dict], ctx: dict) -> tuple[dict, dict]:
    paths, root, block_us = ctx["paths"], ctx["root"], ctx["block_us"]
    per_unit, rows = [], []
    maps = {"real": np.zeros((HEIGHT, WIDTH)), "v2e": np.zeros((HEIGHT, WIDTH))}
    span_s = 0.0
    for u in units:
        seq = u["sequence"]
        aps = ctx["aps"].setdefault(seq, read_aps_timestamps_us(K._data_file(paths, seq)))
        real_store = Path(paths["outputs"]["events"]) / f"mvsec_{seq}_left.h5"
        syn_store = root / u["corpus"] / f"{u['name']}.h5"
        man = json.loads(syn_store.with_suffix(".json").read_text())
        real = read_real(real_store, int(aps[u["aps"][0]]), int(aps[u["aps"][1]]))
        syn = read_syn(syn_store)
        st = F.unit_stats(real, syn, aps, u, block_us, WIDTH, HEIGHT)
        per_unit.append(st)
        for k, e in (("real", real), ("v2e", syn)):
            maps[k] += rate_image(e, st["span_us"])
        span_s += (st["span_us"][1] - st["span_us"][0]) / 1e6
        if real_store not in ctx["real_sha"]:
            ctx["real_sha"][real_store] = _sha256(real_store)
        d = F.distances(F.pool([st]))
        rows.append({"key": u["key"], "sequence": seq, "aps": u["aps"],
                     "stats_window_starts": [int(F.owned_window_starts(u)[0]),
                                             int(F.owned_window_starts(u)[-1])],
                     "n_windows": st["n_windows"], "n_blocks": st["n_blocks"],
                     "span_us": st["span_us"],
                     "real_store": redact_home(str(real_store)),
                     "real_store_sha256": ctx["real_sha"][real_store],
                     "v2e_store_sha256": man["store_sha256"],
                     "v2e_run_manifest_sha256": man["run_manifest_sha256"],
                     "n_events": {"real": st["real"]["n_events"], "v2e": st["v2e"]["n_events"]},
                     "mean_w1_normalised": d["mean_w1_normalised"]})
    pooled = F.pool(per_unit)
    dist = F.distances(pooled)
    jsonable = {k: {**pooled[k], "hist": {s: v.tolist() for s, v in pooled[k]["hist"].items()}}
                for k in ("real", "v2e")}
    doc = {
        "kind": "v2e_fidelity", "tier": "T2",
        "tier_note": "event statistics of v2e(MVSEC APS) vs real DVS of the same sensor and "
                     "windows; simulator-fidelity statistics, not flow accuracy",
        "plan_step": "Phase 3 step 2", "group": name, "dataset": "mvsec",
        "split": units[0]["split"], "sequences": sorted({u["sequence"] for u in units}),
        "profile": units[0]["profile"], "seed": units[0]["seed"], "used_for_choice": False,
        "window_definition": "owned APS dt=1 windows from stats_from_window (warm-up excluded); "
                             "both streams restricted to the owned span; blocks of "
                             f"{block_us // 1000} ms tile the span (sim/fidelity.py, D-033)",
        "stat_definition": {"module": "src/evflow/sim/stats.py", "stats": list(S.STATS),
                            "edges": {k: [float(S.EDGES[k][0]), float(S.EDGES[k][-1]),
                                          len(S.EDGES[k]) - 1] for k in S.STATS},
                            "distance": "W1 on histogram bin centres; normalised = W1 / "
                                        "max(std(real), bin width) (D-027)"},
        "n_windows": pooled["n_windows"], "n_blocks": pooled["n_blocks"],
        "span_s": pooled["span_s"], "real": jsonable["real"], "v2e": jsonable["v2e"],
        "events_per_s_ratio_v2e_over_real": (pooled["v2e"]["events_per_s"]
                                             / pooled["real"]["events_per_s"]),
        "distances": dist, "units": rows, "notes": NOTES,
        "git_sha": ctx["git_sha"], "created_by": "scripts/fidelity_report.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return doc, {k: v / span_s for k, v in maps.items()}


# ---------------------------------------------------------------------------------------------
# figures (static PNG; real = slot 1 blue, v2e = slot 2 orange; one shared scale per panel pair)

LABELS = {"rate_per_window": "events/s per window (log10)", "on_fraction": "ON fraction per window",
          "pixel_rate": "per-pixel rate, active pixels (log10 ev/s)",
          "iei": "per-pixel inter-event interval (log10 µs)",
          "autocorr": "count-image autocorrelation, lags 1–3 px",
          "hot_fraction": "hot-pixel fraction per 500 ms block"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_stats(doc: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), facecolor="#fcfcfb")
    for ax, k in zip(axes.ravel(), S.STATS, strict=True):
        e = S.EDGES[k]
        c = 0.5 * (e[1:] + e[:-1])
        for name, col in (("real", REAL_C), ("v2e", V2E_C)):
            h = np.asarray(doc[name]["hist"][k], float)
            if h.sum():
                ax.step(c, h / h.sum(), where="mid", color=col, linewidth=2, label=name)
        nz = np.nonzero(np.asarray(doc["real"]["hist"][k]) + np.asarray(doc["v2e"]["hist"][k]))[0]
        if nz.size:
            pad = max(2, (nz[-1] - nz[0]) // 10)
            ax.set_xlim(c[max(nz[0] - pad, 0)], c[min(nz[-1] + pad, len(c) - 1)])
        _style(ax)
        ax.set_facecolor("#fcfcfb")
        ax.set_title(LABELS[k], fontsize=9, color=INK, loc="left")
        w = doc["distances"]["w1_normalised"][k]
        ax.text(0.98, 0.95, f"W1/σ = {w:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK2)
        ax.set_ylabel("share of samples", fontsize=8, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{doc['group']}: real DVS vs v2e(APS), profile {doc['profile']} "
                 f"(T2, reported only)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_rate_maps(name: str, maps: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)
    logs = {k: np.log10(v + 1e-3) for k, v in maps.items()}
    vmax = max(float(np.percentile(v, 99.5)) for v in logs.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), facecolor="#fcfcfb")
    for ax, k in zip(axes, ("real", "v2e"), strict=True):
        im = ax.imshow(logs[k], cmap=cmap, vmin=-1, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{k}: mean events/s per pixel (log10)", fontsize=9, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"{name}: per-pixel event rate, shared scale (MVSEC, CC BY-SA 4.0)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", action="append", help="only these groups")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    prof = yaml.safe_load((REPO / "configs/v2e_profiles.yaml").read_text())
    plan = K.load_plan(paths)
    if plan["units"] != K.build_plan()["units"]:
        raise SystemExit("corpus_plan.json differs from the plan rebuilt from configs + splits")
    ctx = {"paths": paths, "root": K.out_root(paths), "aps": {}, "real_sha": {},
           "block_us": int(prof["calibration_grid"]["block_ms"]) * 1000,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                     text=True).stdout.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    summary = []
    for name, units in groups(plan).items():
        if a.group and name not in a.group:
            continue
        doc, maps = run_group(name, units, ctx)
        (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
        if not a.no_figures:
            fig_stats(doc, OUT / "figures" / f"{name}_stats.png")
            fig_rate_maps(name, maps, OUT / "figures" / f"{name}_rate_maps.png")
        row = {"group": name, "file": f"results/fidelity/{name}.json",
               "n_windows": doc["n_windows"],
               "events_per_s": {k: doc[k]["events_per_s"] for k in ("real", "v2e")},
               "events_per_s_ratio_v2e_over_real": doc["events_per_s_ratio_v2e_over_real"],
               "on_fraction_mean": {k: doc[k]["mean"]["on_fraction"] for k in ("real", "v2e")},
               "mean_w1_normalised": doc["distances"]["mean_w1_normalised"],
               "w1_normalised": doc["distances"]["w1_normalised"]}
        summary.append(row)
        print(json.dumps({k: row[k] for k in ("group", "n_windows", "mean_w1_normalised")}))
    if not a.group:
        (OUT / "summary.json").write_text(json.dumps({
            "kind": "v2e_fidelity_summary", "tier": "T2", "plan_step": "Phase 3 step 2",
            "used_for_choice": False, "groups": summary, "notes": NOTES,
            "git_sha": ctx["git_sha"], "created_by": "scripts/fidelity_report.py",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
