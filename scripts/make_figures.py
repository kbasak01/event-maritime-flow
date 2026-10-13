"""Phase 6 figures (plan Phase 6 step 3).

    python scripts/make_figures.py               # EPE figures (JSON only) + flow panels (data)
    python scripts/make_figures.py --bars-only   # JSON-only figures (no data, no GPU)
    python scripts/make_figures.py --readme      # README gallery media (data + GPU, Phase 10)

1. ``results/figures/epe_mvsec_T1.png`` / ``epe_mvsec_T2.png``: per-sequence EPE with 95 % CIs,
   one panel per sequence x dt, arms on the y axis. Numbers come only from
   ``results/eval/aggregate.json`` and the baseline unit JSON (the same records as
   results/TABLES.md); T1 and T2 never share a figure. No imagery, so the files are committed.
2. ``results/figures/flow_panels/*.png``: events / GT / predictions of the seed-1337 runs listed in
   ``configs/evflow.yaml figures`` for the fixed windows there (position in the unit's window
   list, dt = 1, chosen before any prediction was viewed). Flow colour: hue = direction,
   brightness = magnitude, saturated at the window's 99th-percentile GT magnitude (shared by
   GT and predictions); predictions shown inside the sequence's evaluation region only.
   MVSEC imagery, CC BY-SA 4.0: gitignored, attribution in the caption, publication decided in
   Phase 10 (D-008, D-033).
3. ``results/figures/clutter_ct.png`` (Phase 7, D-058): sea-clutter event budget vs the v2e
   contrast-threshold multiplier, from ``results/maritime/clutter.json`` only (T3b, synthetic):
   per-class event share (pixel share dotted) and events per pixel per frame interval, two
   panels, one axis each. No imagery, committed. Written with the JSON-only figures.
4. ``results/figures/maritime_panels/*.png`` (Phase 7 step 6): LaRS frame / events / predicted
   flow / IWE panels; LaRS imagery (CC BY-NC 4.0): gitignored, publication decided in Phase 10
   (D-008). SMD imagery is never drawn (docs/LICENSES.md).
5. ``results/figures/readme/`` (Phase 10, D-065, D-067; committed): README gallery media.
   JSON-only (also written by ``--bars-only``): ``gap_T1.png`` / ``gap_T2.png`` (Table B gaps,
   ``results/eval/paired.json``), ``latency.png`` (``results/latency.json``),
   ``teacher_epe_lars.png`` (T3a, ``results/eval_maritime/summary.json``). Data-needing
   (``--readme``): ``hero_mvsec_outdoor_day1.gif`` (real events / v2e(APS) events / GT /
   predictions, consecutive windows from the fixed outdoor_day1 panel window; no APS frame),
   the six flow panels downsized, the outdoor_day1 fidelity figure, and the two LaRS GIFs
   *without* the source frame (D-065). Per-file size caps from ``configs/evflow.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from evflow.eval import study  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
# dataviz reference palette (light): neutral for baselines, slot 1 blue = trained on real
# events, slot 2 orange = trained on v2e events; identity is also on the y-axis labels.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL, REAL, V2E = "#8a8985", "#2a78d6", "#eb6834"
ROWS_T1 = [("zero_flow", "M0a zero flow"), ("cm_baseline", "M0b contrast max."),
           ("eraft_mvsec20", "R1 E-RAFT (ref. ext.)"), ("real_sup", "real_sup"),
           ("v2e_fixed_sup", "v2e_fixed_sup"), ("v2e_rand_sup", "v2e_rand_sup"),
           ("real_selfsup", "real_selfsup"), ("v2e_selfsup", "v2e_selfsup")]


def _colour(arm: str) -> str:
    if arm in study.BASELINE_MODEL:
        return NEUTRAL
    return REAL if arm.startswith("real") else V2E


def _record(tier: str, arm: str, seq: str, dt: int, agg: dict) -> tuple | None:
    if arm in study.BASELINE_MODEL:
        p = REPO / study.baseline_path(arm, study.stem("mvsec", seq, dt,
                                                      events="v2e" if tier == "T2" else "real"))
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return d["metrics"]["epe"], d["ci95"]["epe"]
    e = agg.get((tier, arm, seq, dt))
    if e is None or not e.get("seeds_complete"):
        return None
    return e["metrics"]["epe"], e["ci95"]["epe"]


def epe_figure(tier: str) -> Path:
    agg = {}
    for e in json.loads((REPO / "results/eval/aggregate.json").read_text())["entries"]:
        if e["representation"] == study.MAIN_REP and e["dataset"] == "mvsec":
            agg[(e["tier"], e["arm"], e["sequence"], e["dt"])] = e
    rows = ROWS_T1 if tier == "T1" else [r for r in ROWS_T1 if r[0] not in
                                          ("cm_baseline", "eraft_mvsec20")]
    seqs, dts = study.MVSEC_SEQS, study.DTS
    fig, axes = plt.subplots(len(dts), len(seqs), figsize=(13, 2.0 + 0.32 * len(rows) * 2),
                             sharey=True, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for i, dt in enumerate(dts):
        for j, seq in enumerate(seqs):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for (arm, _), yy in zip(rows, y, strict=True):
                r = _record(tier, arm, seq, dt, agg)
                if r is None:
                    ax.text(0.02, yy, "n/a", va="center", fontsize=7, color=INK2,
                            transform=ax.get_yaxis_transform())
                    continue
                m, (lo, hi) = r
                ax.plot([lo, hi], [yy, yy], color=_colour(arm), lw=2, solid_capstyle="round")
                ax.plot([m], [yy], "o", ms=6, color=_colour(arm), mec=SURFACE, mew=1.5)
            zero = _record(tier, "zero_flow", seq, dt, agg)
            if zero is not None:
                ax.axvline(zero[0], color=NEUTRAL, lw=1, ls=(0, (3, 3)), zorder=0)
            ax.set_xlim(left=0)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8, length=0)
            ax.set_title(f"{seq}, dt = {dt}", fontsize=9, color=INK, loc="left")
            if i == len(dts) - 1:
                ax.set_xlabel("EPE (px over the window)", fontsize=8, color=INK2)
    axes[0, 0].set_yticks(y, [lbl for _, lbl in rows])
    for i in range(len(dts)):
        axes[i, 0].tick_params(axis="y", labelsize=8, labelcolor=INK)
    events = "real test events" if tier == "T1" else "v2e(APS) test events"
    fig.suptitle(f"MVSEC test, {events} ({tier}): mean EPE with 95 % moving-block bootstrap CI;"
                 " 3-seed arms: seed mean. Dashed line: zero flow.\nGrey = baselines, blue = "
                 "trained on real events, orange = trained on v2e events.", fontsize=9,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"epe_mvsec_{tier}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def flow_rgb(flow: np.ndarray, scale: float, valid: np.ndarray | None = None) -> np.ndarray:
    """(2, H, W) px flow -> RGB in [0, 1]: hue = direction, value = |flow| / scale (clipped)."""
    u, v = flow
    mag = np.hypot(u, v)
    hue = (np.arctan2(-v, -u) / np.pi + 1.0) / 2.0
    hsv = np.stack([hue, np.ones_like(hue), np.clip(mag / max(scale, 1e-6), 0, 1)], -1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    if valid is not None:
        rgb[~valid] = 0.0
    return rgb


def flow_panels() -> list[Path]:
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from run_eval import NetPredictor, load_run

    from evflow.eval.datasets import mvsec_unit
    from evflow.eval.windows import region_mask

    cfg = yaml.safe_load((REPO / "configs/evflow.yaml").read_text())["figures"]
    ev_cfg = yaml.safe_load((REPO / "configs/eval_mvsec.yaml").read_text())
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    runs_root = Path(paths["outputs"]["runs"]) / "train"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = []
    for run_id in cfg["panel_runs"]:
        model, _, man = load_run(runs_root / run_id, device)
        models.append((run_id, man["representation"], model))
    out_dir = FIG / "flow_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for k, spec in enumerate(cfg["panel_windows"]):
        seq, dt = spec["sequence"], int(cfg["panel_dt"])
        region = ev_cfg["regions"][ev_cfg["sequences"][seq]["region"]]
        windows, data, _ = mvsec_unit(paths, seq, dt, split="mvsec_test", region=region)
        w = windows[int(round(spec["position"] * (len(windows) - 1)))]
        ev = data.slicer.get_events(w.t_start_us, w.t_end_us)
        gt, valid = data.gt(w)
        h, wd = data.height, data.width
        img = np.zeros((h, wd))
        np.add.at(img, (ev["y"].astype(int), ev["x"].astype(int)),
                  np.where(ev["p"] > 0, 1.0, -1.0))
        lim = max(np.percentile(np.abs(img[img != 0]), 99), 1.0) if (img != 0).any() else 1.0
        mag = np.hypot(*gt)[valid]
        scale = float(np.percentile(mag, 99)) if mag.size else 1.0
        inside = region_mask(region, h, wd)
        preds = [(rid, NetPredictor(m, rep, h, wd, device)(w, ev)) for rid, rep, m in models]
        fig, axes = plt.subplots(1, 2 + len(preds), figsize=(3.2 * (2 + len(preds)), 3.0))
        axes[0].imshow(img, cmap="gray", vmin=-lim, vmax=lim)
        axes[0].set_title("events (ON +, OFF −)", fontsize=9)
        axes[1].imshow(flow_rgb(gt, scale, valid))
        axes[1].set_title("GT flow (valid px)", fontsize=9)
        for ax, (rid, f) in zip(axes[2:], preds, strict=True):
            ax.imshow(flow_rgb(f, scale, inside))
            ax.set_title(rid.replace("_voxel5_s1337", " (s1337)") + ", eval region", fontsize=9)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"{seq} window {w.index} (dt = {dt}, position {spec['position']}); colour "
                     f"saturates at {scale:.2f} px. T1. {cfg['attribution'].strip()}",
                     fontsize=7, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out = out_dir / f"panel{k}_{seq}_w{w.index}_dt{dt}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outs.append(out)
    return outs


# Phase 7 clutter figure: reference palette slots 1-3 in fixed order (water, sky, obstacle);
# marker shape and direct labels carry class identity as well (never colour alone).
CLASS_STYLE = {"water": ("#2a78d6", "o"), "sky": ("#eb6834", "s"), "obstacle": ("#1baf7a", "^")}


def clutter_figure() -> Path | None:
    src = REPO / "results" / "maritime" / "clutter.json"
    if not src.exists():
        return None
    c = json.loads(src.read_text())
    rows = sorted(c["by_ct"].values(), key=lambda d: d["ct_multiplier"])
    ct = [d["ct_multiplier"] for d in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), facecolor=SURFACE)
    for ax, key, title, ylab in (
            (axes[0], "event_share", "Share of events per class (dotted: share of pixels)",
             "share of the window"),
            (axes[1], "events_per_px", "Events per pixel per frame interval", "events / px")):
        ax.set_facecolor(SURFACE)
        for cls, (col, mk) in CLASS_STYLE.items():
            v = [d["pooled"][cls][key] for d in rows]
            m = [x["mean"] for x in v]
            lo = [x["ci95"][0] if x["ci95"] else x["mean"] for x in v]
            hi = [x["ci95"][1] if x["ci95"] else x["mean"] for x in v]
            ax.fill_between(ct, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(ct, m, color=col, lw=2, marker=mk, ms=8, mec=SURFACE, mew=2, label=cls)
            if key == "event_share":
                ps = rows[0]["pooled"][cls]["pixel_share"]["mean"]
                ax.axhline(ps, color=col, lw=1.2, ls=(0, (1, 2)))
            ax.annotate(cls, (ct[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        if key == "events_per_px":
            ax.set_yscale("log")
        ax.set_xticks(ct, [f"×{x:g}" for x in ct])
        ax.set_xlim(ct[0] - 0.1, ct[-1] + 0.3)
        ax.set_xlabel("v2e contrast threshold (× calibrated)", color=INK2, fontsize=8)
        ax.set_ylabel(ylab, color=INK2, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"T3b, synthetic: v2e(LaRS) keyframe windows of {c['n_clips']} lars_eval "
                 "clips; LaRS frame rate assumed 30 fps; 95 % cluster-bootstrap CI over clips",
                 fontsize=8, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "clutter_ct.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


LARS_ATTRIBUTION = ("LaRS: Žust, Perš, Kristan, ICCV 2023, CC BY-NC 4.0 (non-commercial); "
                    "frames resized to 640×360 as v2e did")


def maritime_panels(n_clips: int = 2, run_id: str = "maritime_teacher_voxel5_s1337",
                    source_frame: bool = True, out_dir: Path | None = None) -> list:
    """GIF per LaRS clip: frame / events / predicted flow / IWE with the predicted flow, for every
    window after the warm-up. Clips = the first ``n_clips`` lars_eval corpus clips (fixed before
    any prediction was viewed); model = the maritime_teacher run. Gitignored (D-008).
    ``source_frame=False`` drops the LaRS frame column (README gallery, D-065: derived media
    only, the original image is not shown)."""
    import cv2
    import imageio.v2 as imageio
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from run_eval import NetPredictor, load_run

    from evflow.eval import maritime as em
    from evflow.flow.iwe import iwe
    from evflow.io import lars as lars_io
    from evflow.io.frames import resize_like_v2e

    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, man = load_run(Path(paths["outputs"]["runs"]) / "train" / run_id, device)
    segs = em.segments(paths, "lars", "lars_eval_all")[:n_clips]
    from evflow.sim import maritime_corpus as mc
    plan = {u["key"]: u for u in mc.load_plan(paths)["units"]}
    out_dir = out_dir or FIG / "maritime_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for seg in segs:
        u = plan[seg.key]
        prefix, first, _ = u["clip"].rsplit(":", 2)
        files = lars_io.clip_frames(paths["lars"]["root"], u["subset"], prefix, int(first),
                                    int(u["clip"].rsplit(":", 1)[1]))
        net = NetPredictor(model, man["representation"], 360, 640, device)
        frames = []
        for w, k in zip(seg.windows, seg.local_k, strict=True):
            ev = seg.slicer.get_events(w.t_start_us, w.t_end_us)
            f = net(w, ev)
            rgb = (cv2.cvtColor(resize_like_v2e(cv2.imread(str(files[k + 1])), (640, 360)),
                                cv2.COLOR_BGR2RGB) if source_frame else None)
            img = np.zeros((360, 640))
            np.add.at(img, (ev["y"].astype(int), ev["x"].astype(int)),
                      np.where(ev["p"] > 0, 1.0, -1.0))
            lim = max(np.percentile(np.abs(img[img != 0]), 99), 1.0) if (img != 0).any() else 1
            xi = torch.from_numpy(ev["x"].astype(np.int64))
            yi = torch.from_numpy(ev["y"].astype(np.int64))
            ti = torch.from_numpy(ev["t"].astype(np.int64))
            im = iwe(xi, yi, ti, torch.from_numpy(f), t_range=(w.t_start_us, w.t_end_us),
                     t_ref=1.0, height=360, width=640, interpolation="bilinear").numpy()
            scale = max(float(np.percentile(np.hypot(*f), 99)), 0.5)
            fig, axes = plt.subplots(1, 3 + source_frame,
                                     figsize=(13 if source_frame else 9.9, 2.5))
            if source_frame:
                axes[0].imshow(rgb)
                axes[0].set_title(f"frame {k + 1}", fontsize=8)
            ax_ev, ax_fl, ax_iwe = axes[int(source_frame):]
            ax_ev.imshow(img, cmap="gray", vmin=-lim, vmax=lim)
            ax_ev.set_title("v2e events (ON +, OFF −)", fontsize=8)
            ax_fl.imshow(flow_rgb(f, scale))
            ax_fl.set_title(f"predicted flow ({run_id.split('_voxel')[0]}), "
                            f"sat. {scale:.1f} px", fontsize=8)
            ax_iwe.imshow(im, cmap="gray_r", vmax=max(np.percentile(im, 99.5), 1e-6))
            ax_iwe.set_title("IWE with predicted flow (t_ref = 1)", fontsize=8)
            for ax in axes:
                ax.set_axis_off()
            attribution = (LARS_ATTRIBUTION if source_frame else LARS_ATTRIBUTION.split(
                "; frames")[0] + "; derived (events / flow / IWE only), source frame not shown")
            fig.suptitle(f"{seg.key}, window {k} -> {k + 1} (one frame interval). T3, synthetic "
                         f"events, no ground truth.\n{attribution}", fontsize=6, x=0.01,
                         ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.84))
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        out = out_dir / (f"{seg.key.split('/')[-1]}.gif" if source_frame
                         else f"maritime_{seg.key.split('/')[-1]}.gif")
        imageio.mimsave(out, frames, duration=0.4, loop=0)
        outs.append(out)
    return outs


# --- Phase 10: README gallery (D-065, D-067) -----------------------------------------------------
README_DIR = FIG / "readme"
MARITIME_C = "#1baf7a"     # reference palette slot 3 (after blue = real, orange = v2e)


def _readme_cfg() -> dict:
    return yaml.safe_load((REPO / "configs/evflow.yaml").read_text())["figures"]["readme"]


def _cap(path: Path) -> Path:
    """Enforce the per-file size cap of configs/evflow.yaml figures.readme.max_bytes."""
    cap = _readme_cfg()["max_bytes"][path.suffix.lstrip(".")]
    size = path.stat().st_size
    if size > cap:
        raise SystemExit(f"{path.relative_to(REPO)}: {size} bytes > cap {cap}")
    return path


def _clean(ax) -> None:
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


GAP_COMPS = (("gap_v2e_fixed_sup", "v2e_fixed_sup − real_sup", "3 seeds per arm"),
             ("gap_v2e_rand_sup", "v2e_rand_sup − real_sup", "3 seeds per arm"),
             ("gap_selfsup", "v2e_selfsup − real_selfsup", "1 seed per arm"))


def gap_figure(tier: str) -> Path:
    """Headline: Table B EPE gaps (v2e-trained minus real-trained, same windows) per sequence and
    pooled, 95 % paired-bootstrap CI (windows) and range over single-seed pairings. One tier per
    figure; numbers only from results/eval/paired.json (the records of Table B)."""
    paired = json.loads((REPO / "results/eval/paired.json").read_text())
    P = {(e["comparison"], e["sequence"], e.get("dt")): e for e in paired["entries"]
         if e["table"] == "B" and e["tier"] == tier}
    rows = list(study.MVSEC_SEQS) + ["pooled"]
    dts = study.DTS
    fig, axes = plt.subplots(len(dts), len(GAP_COMPS), figsize=(12.5, 5.4), sharey=True,
                             sharex="row", facecolor=SURFACE)
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= 0.4                                    # pooled row set apart
    for i, dt in enumerate(dts):
        for j, (comp, label, seeds) in enumerate(GAP_COMPS):
            ax = axes[i, j]
            _clean(ax)
            for seq, yy in zip(rows, y, strict=True):
                e = P.get((comp, seq, dt))
                if e is None or e["estimate"] is None:
                    ax.text(0.02, yy, "n/a", va="center", fontsize=7, color=INK2,
                            transform=ax.get_yaxis_transform())
                    continue
                r = (e.get("seed_pairings") or {}).get("range")
                if r:
                    ax.plot(r, [yy, yy], color=V2E, alpha=0.25, lw=7, solid_capstyle="round")
                lo, hi = e["ci95"]
                ax.plot([lo, hi], [yy, yy], color=V2E, lw=2, solid_capstyle="round")
                ax.plot([e["estimate"]], [yy], "o", ms=7, color=V2E, mec=SURFACE, mew=1.5)
            ax.axvline(0, color=INK2, lw=1)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{label}, dt = {dt}  ({seeds})", fontsize=9, color=INK, loc="left")
            if i == len(dts) - 1:
                ax.set_xlabel("EPE gap (px over the window); > 0: v2e training is worse",
                              fontsize=8, color=INK2)
    floor = tier == "T1"         # D-046 is a T1 observation (Table A-T1)
    labels = [s + (" †" if floor and s.startswith("indoor") else "") for s in rows[:-1]] + [
        "pooled (mean of 4)"]
    axes[0, 0].set_yticks(y, labels)
    for i in range(len(dts)):
        axes[i, 0].tick_params(axis="y", labelsize=8, labelcolor=INK)
    events = "real test events" if tier == "T1" else "v2e(APS) test events"
    fig.suptitle(
        f"Matched-content simulator gap, MVSEC test, {events} ({tier}). Same network, same GT "
        "labels; only the training events differ.\nDot + line: estimate and 95 % paired "
        "moving-block bootstrap CI over windows. Pale band: range over single-seed pairings "
        "(between-seed spread, not in the CI)."
        + ("\n† indoor: supervised arms at or near the zero-flow floor at dt = 1 (D-046); a "
           "negative gap there is not a v2e advantage." if floor else ""),
        fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()  # accounts for the suptitle
    README_DIR.mkdir(parents=True, exist_ok=True)
    out = README_DIR / f"gap_{tier}.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return _cap(out)


BACKENDS = (("pytorch_eager_fp32", "PyTorch eager fp32", REAL, None),
            ("ort_cuda_fp32", "ORT CUDA EP fp32", V2E, None),
            ("ort_cuda_fp16", "ORT CUDA EP fp16 ✗", V2E, "///"),
            ("ort_trt_fp16", "ORT TensorRT EP fp16 ✗", MARITIME_C, "///"))


def latency_figure() -> Path | None:
    """J2 network p50 per backend x input and J3 end-to-end p50 vs events per window at 346×260,
    from results/latency.json only. Hatched = fp16 graph/builder that failed parity (D-064)."""
    src = REPO / "results/latency.json"
    if not src.exists():
        return None
    d = json.loads(src.read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.0), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    sizes = []
    for n in d["network"]:
        if (n["height"], n["width"]) not in sizes:
            sizes.append((n["height"], n["width"]))
    net = {(n["backend"], n["height"], n["width"]): n for n in d["network"]}
    bw = 0.8 / len(BACKENDS)
    for k, (be, lab, col, hatch) in enumerate(BACKENDS):
        xs, vs = [], []
        for i, (h, w) in enumerate(sizes):
            n = net.get((be, h, w))
            if n is not None:
                xs.append(i - 0.4 + bw * (k + 0.5))
                vs.append(n["compute"]["p50_ms"])
        bars = a1.bar(xs, vs, width=bw * 0.9, color=col if hatch is None else SURFACE,
                      edgecolor=col, hatch=hatch, lw=1.5, label=lab)
        for b, v in zip(bars, vs, strict=True):
            a1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, color=INK2)
    a1.set_xticks(range(len(sizes)), [f"{w}×{h}" for h, w in sizes])
    a1.set_ylabel("network forward p50 (ms)", fontsize=8, color=INK2)
    a1.set_title("Network forward (compute only), batch 1", fontsize=9, color=INK, loc="left")
    _clean(a1)
    a1.grid(axis="y", color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    e2e = [e for e in d["end_to_end"] if (e["height"], e["width"]) == (260, 346)]
    for be, lab, col, hatch in BACKENDS:
        pts = sorted((e["events"], e["total"]["p50_ms"]) for e in e2e if e["backend"] == be)
        if not pts:
            continue
        a2.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2,
                ls="-" if hatch is None else (0, (4, 2)), marker="o", ms=7, mec=SURFACE,
                mew=1.5, label=lab)
    a2.set_xscale("log")
    a2.set_xlabel("events per window", fontsize=8, color=INK2)
    a2.set_ylabel("end-to-end p50 (ms)", fontsize=8, color=INK2)
    a2.set_ylim(bottom=0)
    a2.set_title("End-to-end events → flow, 346×260 (ORT rows composed, D-063)", fontsize=9,
                 color=INK, loc="left")
    _clean(a2)
    a2.grid(axis="y", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Hardware timing of the M1 network (no evidence tier). RTX A4000 stands in for "
                 "an embedded target; not event-sensor latency.\n✗ / hatched / dashed = fp16: "
                 "failed the parity tolerance (D-064), not a deployable configuration.",
                 fontsize=8, color=INK, x=0.01, ha="left")
    fig.tight_layout()  # accounts for the suptitle
    README_DIR.mkdir(parents=True, exist_ok=True)
    out = README_DIR / "latency.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return _cap(out)


T3_MODELS = (("zero_flow", "M0a zero flow", NEUTRAL), ("cm_baseline", "M0b contrast max.", NEUTRAL),
             ("eraft_dsec", "R1 E-RAFT DSEC (ref. ext.)", NEUTRAL),
             ("real_sup", "real_sup", REAL), ("real_selfsup", "real_selfsup", REAL),
             ("v2e_fixed_sup", "v2e_fixed_sup", V2E), ("v2e_rand_sup", "v2e_rand_sup", V2E),
             ("v2e_selfsup", "v2e_selfsup", V2E),
             ("maritime_selfsup", "maritime_selfsup", MARITIME_C),
             ("maritime_teacher", "maritime_teacher *", MARITIME_C))


def teacher_figure() -> Path | None:
    """T3a teacher-EPE on the LaRS keyframe windows, sliced by class, from
    results/eval_maritime/summary.json only (Table F-T3a)."""
    src = REPO / "results/eval_maritime/summary.json"
    if not src.exists():
        return None
    s = json.loads(src.read_text())
    E = {(e["arm"], e["slice"]): e for e in s["entries"]
         if e["tier"] == "T3a" and e["unit"] == "lars_eval_kf" and e["metric"] == "teacher_epe"}
    slices = ("all", "water", "sky", "obstacle")
    fig, axes = plt.subplots(1, len(slices), figsize=(12.5, 3.9), sharey=True, sharex=True,
                             facecolor=SURFACE)
    y = np.arange(len(T3_MODELS))[::-1]
    for ax, sl in zip(axes, slices, strict=True):
        _clean(ax)
        for (arm, _, col), yy in zip(T3_MODELS, y, strict=True):
            e = E.get((arm, sl))
            if e is None:
                continue
            ax.plot(e["ci95"], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot([e["mean"]], [yy], "o", ms=7, color=col, mec=SURFACE, mew=1.5)
        n = (E.get(("zero_flow", sl)) or {}).get("n_windows")
        ax.set_title(f"keyframes: {sl} (n = {n})", fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("teacher-EPE (px per frame interval)", fontsize=8, color=INK2)
    axes[0].set_xlim(0, 1.05 * max(e["ci95"][1] for e in E.values()))
    axes[0].set_yticks(y, [lab for _, lab, _ in T3_MODELS])
    axes[0].tick_params(axis="y", labelsize=8, labelcolor=INK)
    fig.suptitle("T3a, synthetic v2e(LaRS) events: agreement with a frame-based RAFT teacher, "
                 "not accuracy (the teacher is not ground truth). 95 % cluster-bootstrap CI "
                 "over clips; LaRS frame rate assumed 30 fps.\nGrey = baselines, blue = trained "
                 "on real MVSEC events, orange = trained on v2e(MVSEC), green = trained on "
                 "v2e(SMD + LaRS). * trained against this teacher.",
                 fontsize=8, color=INK, x=0.01, ha="left")
    fig.tight_layout()  # accounts for the suptitle
    README_DIR.mkdir(parents=True, exist_ok=True)
    out = README_DIR / "teacher_epe_lars.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return _cap(out)


def _fig_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _downsize(src: Path, dst: Path) -> Path:
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w = _readme_cfg()["max_width_px"]
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)
    return _cap(dst)


def _event_image(ev: dict, h: int, w: int) -> tuple[np.ndarray, float]:
    img = np.zeros((h, w))
    np.add.at(img, (ev["y"].astype(int), ev["x"].astype(int)), np.where(ev["p"] > 0, 1.0, -1.0))
    lim = max(np.percentile(np.abs(img[img != 0]), 99), 1.0) if (img != 0).any() else 1.0
    return img, lim


def hero_gif() -> Path:
    """Consecutive dt = 1 windows of outdoor_day1 from the fixed panel position: real events,
    v2e(APS) events of the same window, GT, and the panel_runs' predictions on the real events
    (T1). No APS frame (D-008). Colour scale fixed over the clip (99th pct of GT magnitude)."""
    import imageio.v2 as imageio
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from run_eval import NetPredictor, load_run

    from evflow.eval.datasets import mvsec_unit
    from evflow.eval.windows import region_mask

    fcfg = yaml.safe_load((REPO / "configs/evflow.yaml").read_text())["figures"]
    hcfg = fcfg["readme"]["hero"]
    ev_cfg = yaml.safe_load((REPO / "configs/eval_mvsec.yaml").read_text())
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seq = hcfg["sequence"]
    region = ev_cfg["regions"][ev_cfg["sequences"][seq]["region"]]
    windows, real, _ = mvsec_unit(paths, seq, 1, split="mvsec_test", region=region)
    vwin, v2e, _ = mvsec_unit(paths, seq, 1, split="mvsec_test", region=region, events="v2e")
    vidx = {w.index: w for w in vwin}
    k0 = int(round(hcfg["position"] * (len(windows) - 1)))
    clip = windows[k0:k0 + int(hcfg["n_windows"])]
    h, wd = real.height, real.width
    inside = region_mask(region, h, wd)
    preds = []
    for run_id in fcfg["panel_runs"]:
        model, _, man = load_run(Path(paths["outputs"]["runs"]) / "train" / run_id, device)
        preds.append((run_id, NetPredictor(model, man["representation"], h, wd, device)))
    data = []
    for w in clip:
        ev = real.slicer.get_events(w.t_start_us, w.t_end_us)
        vw = vidx[w.index]
        vev = v2e.slicer.get_events(vw.t_start_us, vw.t_end_us)
        gt, valid = real.gt(w)
        data.append((w, ev, vev, gt, valid, [(rid, p(w, ev)) for rid, p in preds]))
    mags = np.concatenate([np.hypot(*d[3])[d[4]] for d in data])
    scale = float(np.percentile(mags, 99)) if mags.size else 1.0
    yy, xx = np.mgrid[-1:1:121j, -1:1:121j]
    wheel = flow_rgb(np.stack([xx, yy]) * scale, scale, np.hypot(xx, yy) <= 1)
    frames = []
    for w, ev, vev, gt, valid, pr in data:
        fig, grid = plt.subplots(2, 3, figsize=(10.5, 6.0), facecolor="white")
        axes = [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1]]
        for ax, e, title in ((axes[0], ev, "real DVS events"),
                             (axes[1], vev, "v2e(APS) events, same window")):
            img, lim = _event_image(e, h, wd)
            ax.imshow(img, cmap="gray", vmin=-lim, vmax=lim)
            ax.set_title(title, fontsize=9)
        axes[2].imshow(flow_rgb(gt, scale, valid))
        axes[2].set_title("ground-truth flow (valid px)", fontsize=9)
        for ax, (rid, f) in zip(axes[3:], pr, strict=True):
            ax.imshow(flow_rgb(f, scale, inside))
            ax.set_title(rid.split("_voxel")[0] + " on real events", fontsize=9)
        grid[1, 2].imshow(wheel)
        grid[1, 2].set_title("flow colour key (hue = direction)", fontsize=9)
        for ax in grid.ravel():
            ax.set_axis_off()
        fig.suptitle(f"MVSEC {seq}, window {w.index} (dt = 1 APS frame), {len(clip)} "
                     f"consecutive windows from the fixed panel position {hcfg['position']}; "
                     f"colour saturates at {scale:.2f} px.\nPredictions: T1 (real events, real "
                     "GT), seed 1337, shown in the evaluation region; no APS frame shown.\n"
                     f"{fcfg['attribution'].strip()}", fontsize=7, x=0.01, ha="left")
        fig.tight_layout()
        frames.append(_fig_rgb(fig))
        plt.close(fig)
    from PIL import Image
    wmax = _readme_cfg()["max_width_px"]
    pil = [Image.fromarray(f) for f in frames]
    if pil[0].width > wmax:
        size = (wmax, round(pil[0].height * wmax / pil[0].width))
        pil = [p.resize(size, Image.LANCZOS) for p in pil]
    README_DIR.mkdir(parents=True, exist_ok=True)
    out = README_DIR / f"hero_mvsec_{seq}.gif"
    imageio.mimsave(out, [np.asarray(p) for p in pil], duration=float(hcfg["frame_s"]), loop=0)
    return _cap(out)


def readme_media() -> list[Path]:
    """Data-needing README media: hero GIF, the six flow panels and the fidelity figure
    (downsized copies of the Phase 3/6 outputs, regenerated first if absent), the LaRS GIFs
    without source frames (D-065)."""
    outs = [hero_gif()]
    panels = sorted((FIG / "flow_panels").glob("panel*.png"))
    if len(panels) != len(yaml.safe_load((REPO / "configs/evflow.yaml").read_text())
                          ["figures"]["panel_windows"]):
        panels = flow_panels()
    outs += [_downsize(p, README_DIR / f"flow_{p.name}") for p in panels]
    fid = REPO / _readme_cfg()["fidelity_figure"]
    if not fid.exists():
        raise SystemExit(f"{fid.relative_to(REPO)} missing: run scripts/fidelity_report.py")
    outs.append(_downsize(fid, README_DIR / "fidelity_outdoor_day1.png"))
    for g in maritime_panels(source_frame=False, out_dir=README_DIR):
        outs.append(_cap(g))
    return outs


MEDIA_LICENCE = {     # README media (D-065, D-067): licence by source dataset
    "MVSEC": ("[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)",
              "Derived from MVSEC (Zhu, Thakur, Özaslan, Pfrommer, Kumar, Daniilidis, RA-L "
              "2018; ground-truth flow: Zhu, Yuan, Chaney, Daniilidis, RSS 2018); shared under "
              "the same licence."),
    "LaRS": ("[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial",
             "Derived from LaRS (Žust, Perš, Kristan, ICCV 2023): v2e events, predicted flow and "
             "warped events only; no source frame (D-065)."),
    "LaRS+SMD": ("aggregate metrics, no imagery; non-commercial context (LaRS CC BY-NC 4.0)",
                 "Computed on v2e events from LaRS (Žust et al., ICCV 2023) and SMD (Prasad et "
                 "al., T-ITS 2017); no dataset imagery."),
    "none": ("MIT (repository licence)", "Hardware timing; no dataset content."),
}


def _media_source(name: str) -> str:
    if name.startswith("maritime_lars"):
        return "LaRS"
    if name.startswith("teacher_epe"):
        return "LaRS+SMD"
    if name.startswith("latency"):
        return "none"
    return "MVSEC"     # hero, flow panels, fidelity, gap plots


def media_licence() -> Path:
    """results/figures/readme/LICENSE.md: per-file licence of the README media (reviewer m1)."""
    README_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(README_DIR.iterdir()):
        if f.suffix in (".png", ".gif"):
            lic, att = MEDIA_LICENCE[_media_source(f.name)]
            rows.append(f"| `{f.name}` | {lic} | {att} |")
    extra = [("clutter_ct.png", "LaRS+SMD"), ("epe_mvsec_T1.png", "MVSEC"),
             ("epe_mvsec_T2.png", "MVSEC")]
    for name, src in extra:
        lic, att = MEDIA_LICENCE[src]
        rows.append(f"| `../{name}` | {lic} | {att} |")
    text = "\n".join([
        "# Licences of the README media", "",
        "Generated by `scripts/make_figures.py`; do not edit by hand. These files are **not** "
        "covered by the repository's MIT licence (which covers `src/`, `tests/`, `scripts/`): "
        "each carries the licence of the dataset it is derived from. No raw dataset content "
        "(frames, events, ground truth) is included; see `docs/LICENSES.md`.", "",
        "| file | licence | attribution |", "|---|---|---|", *rows, ""])
    out = README_DIR / "LICENSE.md"
    out.write_text(text)
    return out


def readme_json_figures() -> list[Path]:
    outs = [gap_figure(t) for t in ("T1", "T2")]
    outs += [p for p in (latency_figure(), teacher_figure()) if p is not None]
    return outs + [media_licence()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bars-only", action="store_true")
    ap.add_argument("--maritime-panels-only", action="store_true",
                    help="LaRS qualitative GIFs only (needs data and the maritime runs)")
    ap.add_argument("--readme", action="store_true",
                    help="README gallery media only (needs data, runs and a GPU; Phase 10)")
    args = ap.parse_args(argv)
    if args.maritime_panels_only:
        for p in maritime_panels():
            print(f"wrote {p.relative_to(REPO)}")
        return 0
    if args.readme:
        for p in readme_json_figures() + readme_media():
            print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size} bytes)")
        return 0
    for tier in ("T1", "T2"):
        print(f"wrote {epe_figure(tier).relative_to(REPO)}")
    cf = clutter_figure()
    print(f"wrote {cf.relative_to(REPO)}" if cf else "clutter figure: results/maritime/"
          "clutter.json not present")
    for p in readme_json_figures():
        print(f"wrote {p.relative_to(REPO)}")
    if not args.bars_only:
        for p in flow_panels():
            print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
