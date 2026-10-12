"""Phase 6 figures (plan Phase 6 step 3).

    python scripts/make_figures.py               # EPE figures (JSON only) + flow panels (data)
    python scripts/make_figures.py --bars-only   # EPE figures only (no data, no GPU)

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


def maritime_panels(n_clips: int = 2, run_id: str = "maritime_teacher_voxel5_s1337") -> list:
    """GIF per LaRS clip: frame / events / predicted flow / IWE with the predicted flow, for every
    window after the warm-up. Clips = the first ``n_clips`` lars_eval corpus clips (fixed before
    any prediction was viewed); model = the maritime_teacher run. Gitignored (D-008)."""
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
    out_dir = FIG / "maritime_panels"
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
            rgb = cv2.cvtColor(resize_like_v2e(cv2.imread(str(files[k + 1])), (640, 360)),
                               cv2.COLOR_BGR2RGB)
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
            fig, axes = plt.subplots(1, 4, figsize=(13, 2.5))
            axes[0].imshow(rgb)
            axes[0].set_title(f"frame {k + 1}", fontsize=8)
            axes[1].imshow(img, cmap="gray", vmin=-lim, vmax=lim)
            axes[1].set_title("v2e events (ON +, OFF −)", fontsize=8)
            axes[2].imshow(flow_rgb(f, scale))
            axes[2].set_title(f"predicted flow ({run_id.split('_voxel')[0]}), "
                              f"sat. {scale:.1f} px", fontsize=8)
            axes[3].imshow(im, cmap="gray_r", vmax=max(np.percentile(im, 99.5), 1e-6))
            axes[3].set_title("IWE with predicted flow (t_ref = 1)", fontsize=8)
            for ax in axes:
                ax.set_axis_off()
            fig.suptitle(f"{seg.key}, window {k} -> {k + 1} (one frame interval). T3, synthetic "
                         f"events, no ground truth. {LARS_ATTRIBUTION}", fontsize=6, x=0.01,
                         ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        out = out_dir / f"{seg.key.split('/')[-1]}.gif"
        imageio.mimsave(out, frames, duration=0.4, loop=0)
        outs.append(out)
    return outs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bars-only", action="store_true")
    ap.add_argument("--maritime-panels-only", action="store_true",
                    help="LaRS qualitative GIFs only (needs data and the maritime runs)")
    args = ap.parse_args(argv)
    if args.maritime_panels_only:
        for p in maritime_panels():
            print(f"wrote {p.relative_to(REPO)}")
        return 0
    for tier in ("T1", "T2"):
        print(f"wrote {epe_figure(tier).relative_to(REPO)}")
    cf = clutter_figure()
    print(f"wrote {cf.relative_to(REPO)}" if cf else "clutter figure: results/maritime/"
          "clutter.json not present")
    if not args.bars_only:
        for p in flow_panels():
            print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
