"""Stage the Phase 10 release assets and write their checksums (plan Phase 10; D-066).

    python scripts/make_release.py            # stage release/ (gitignored), write docs/RELEASE.md
    python scripts/make_release.py --check    # verify release/ against docs/SHA256SUMS.txt

Assets (``configs/evflow.yaml`` ``release``): the selected checkpoint (``best.pt``) of each listed
MVSEC run as ``m1_<run>.pt``, verified against its ``selection.json`` ``checkpoint_sha256``; the
fp32 ONNX graphs of the Phase 8 primary checkpoint, verified against their export sidecar;
``run_manifests.json`` (manifest, selection and ONNX sidecars of every shipped file, absolute
paths made repo-relative; Phase 9 auditor m-3); ``WEIGHTS_LICENSE.md``; ``SHA256SUMS.txt``
(``sha256sum -c`` format). No event data, frames, GT, flow arrays or figures: any such file
aborts the script (checklist G2). ``docs/RELEASE.md`` and ``docs/SHA256SUMS.txt`` are the
committed record of what was staged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "release"
ALLOWED = {".pt", ".onnx", ".json", ".md", ".txt"}


class ReleaseError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_asset_name(name: str) -> None:
    """G2: only weights, graphs and text records leave the machine."""
    if Path(name).suffix not in ALLOWED:
        raise ReleaseError(f"{name}: extension not allowed in a release ({sorted(ALLOWED)})")


def relativise(obj, repo: Path, home: Path):
    """Absolute paths in manifests -> repo-relative (or ~/...) strings, recursively."""
    if isinstance(obj, dict):
        return {k: relativise(v, repo, home) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativise(v, repo, home) for v in obj]
    if isinstance(obj, str):
        s = obj.replace(str(repo) + "/", "").replace(str(repo), ".")
        return s.replace(str(home) + "/", "~/")
    return obj


def sums_text(files: list[Path]) -> str:
    return "".join(f"{sha256(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name))


WEIGHTS_LICENSE = """# Licence of the released weights

The checkpoints (`m1_*.pt`) and ONNX graphs (`m1_*_fp32.onnx`) in this release were trained on
MVSEC (Zhu, Thakur, Özaslan, Pfrommer, Kumar, Daniilidis, "The Multi Vehicle Stereo Event Camera
Dataset", IEEE RA-L 3(3):2032-2039, 2018; ground-truth flow: Zhu, Yuan, Chaney, Daniilidis,
"EV-FlowNet", RSS 2018), which is licensed CC BY-SA 4.0. The weights are therefore released under
the **Creative Commons Attribution-ShareAlike 4.0 International** licence
(https://creativecommons.org/licenses/by-sa/4.0/), with the attribution above. The `v2e_*`
weights were trained on events simulated with v2e (Hu, Liu, Delbruck, CVPRW 2021; MIT) from
MVSEC APS frames and supervised by MVSEC ground truth.

The code that produced them is MIT (repository `LICENSE`). No dataset content (events, frames,
ground truth, v2e outputs) is part of this release. No model trained on LaRS or SMD is released.
fp16 graphs are not released: they failed the parity tolerance (docs/DECISIONS.md D-064).
"""


def stage(cfg: dict, runs_root: Path) -> list[dict]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    home = Path.home()
    rows, records = [], {"kind": "release_run_manifests", "runs": {}, "onnx": {}}
    for run in cfg["runs"]:
        rd = runs_root / "train" / run
        sel = json.loads((rd / "selection.json").read_text())
        if sel.get("split") != "mvsec_tune":
            raise ReleaseError(f"{run}: selection split {sel.get('split')} is not mvsec_tune")
        src = rd / sel["checkpoint"]
        digest = sha256(src)
        if digest != sel["checkpoint_sha256"]:
            raise ReleaseError(f"{run}: {src.name} sha256 differs from selection.json")
        dst = STAGE / f"m1_{run}.pt"
        shutil.copy2(src, dst)
        records["runs"][run] = {"selection": relativise(sel, REPO, home),
                                "manifest": relativise(json.loads(
                                    (rd / "manifest.json").read_text()), REPO, home)}
        rows.append({"name": dst.name, "run": run, "arm": sel["arm"], "seed": sel["seed"],
                     "kind": f"checkpoint (epoch {sel['selected_epoch']}, selected on "
                             f"{sel['split']})", "licence": cfg["weights_licence"]})
    ed = runs_root / "export" / cfg["onnx_run"]
    for name in cfg["onnx"]:
        side = json.loads((ed / name.replace(".onnx", ".json")).read_text())
        if side.get("precision") != "fp32":
            raise ReleaseError(f"{name}: only fp32 graphs are released (D-064)")
        if sha256(ed / name) != side["onnx_sha256"]:
            raise ReleaseError(f"{name}: sha256 differs from its export sidecar")
        if side["checkpoint_sha256"] != records["runs"][cfg["onnx_run"]]["selection"][
                "checkpoint_sha256"]:
            raise ReleaseError(f"{name}: exported from a different checkpoint")
        dst = STAGE / f"m1_{cfg['onnx_run']}_{name.removeprefix('m1_')}"
        shutil.copy2(ed / name, dst)
        records["onnx"][dst.name] = relativise(side, REPO, home)
        shape = side["input"]["shape"]
        rows.append({"name": dst.name, "run": cfg["onnx_run"], "arm": side["arm"],
                     "seed": side["seed"], "kind": f"ONNX opset {side['opset']} fp32, input "
                     f"{'×'.join(map(str, shape))}", "licence": cfg["weights_licence"]})
    text = json.dumps(records, indent=1, sort_keys=True)
    if str(home) in text:
        raise ReleaseError("run_manifests.json still holds an absolute home path")
    (STAGE / "run_manifests.json").write_text(text + "\n")
    (STAGE / "WEIGHTS_LICENSE.md").write_text(WEIGHTS_LICENSE)
    rows += [{"name": "run_manifests.json", "run": "all", "arm": "–", "seed": "–",
              "kind": "run manifests, selections, ONNX sidecars (paths repo-relative)",
              "licence": "MIT"},
             {"name": "WEIGHTS_LICENSE.md", "run": "–", "arm": "–", "seed": "–",
              "kind": "licence of the weights", "licence": "–"}]
    for p in STAGE.iterdir():
        check_asset_name(p.name)
    files = sorted(STAGE.iterdir())
    (STAGE / "SHA256SUMS.txt").write_text(sums_text(files))
    for r in rows:
        p = STAGE / r["name"]
        r["bytes"], r["sha256"] = p.stat().st_size, sha256(p)
    return rows


def release_md(cfg: dict, rows: list[dict]) -> str:
    total = sum(r["bytes"] for r in rows)
    L = ["# Release assets", "",
         f"Generated by `scripts/make_release.py` (D-066) for tag `{cfg['tag']}`; do not edit "
         "by hand. The files are staged in the gitignored `release/` directory and uploaded "
         "to the GitHub release; checksums are in `docs/SHA256SUMS.txt` (also uploaded, "
         "`sha256sum -c SHA256SUMS.txt`).", "",
         "Contents: the selected `best.pt` of every main (voxel5) MVSEC run behind Tables A, "
         "B and D (Table C's other representations are not released; epoch chosen on "
         "`mvsec_tune` only; SHA-256 equal to the run's `selection.json`), the fp32 ONNX "
         "graphs of the Phase 8 primary checkpoint (D-063; fp16 graphs failed parity, "
         "D-064, not released), the run manifests, and the weights licence. **No event data, "
         "frames, ground truth, flow arrays or figures** (checklist G2). No maritime model is "
         "released (LaRS CC BY-NC 4.0, SMD without a licence).", "",
         "| file | run | arm | seed | content | licence | bytes | sha256 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['name']}` | {r['run']} | {r['arm']} | {r['seed']} | {r['kind']} | "
                 f"{r['licence']} | {r['bytes']} | `{r['sha256']}` |")
    L += ["", f"Total: {len(rows)} files, {total} bytes (plus `SHA256SUMS.txt`).", "",
          "Load a checkpoint: `evflow.models` EVFlowNet with `in_channels` / `base_channels` "
          "from the checkpoint dict (`scripts/run_eval.py` `load_run`); input = voxel grid, 5 "
          "bins, normalised as in `configs/evflow.yaml` `representations`, padded to a "
          "multiple of 16. The ONNX graphs take the same voxel tensor (representation built "
          "outside the graph, D-063).", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify release/ against docs/SHA256SUMS.txt")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((REPO / "configs/evflow.yaml").read_text())["release"]
    sums = REPO / "docs/SHA256SUMS.txt"
    if args.check:
        expected = sums.read_text()
        got = sums_text([p for p in STAGE.iterdir() if p.name != "SHA256SUMS.txt"])
        ok = got == expected and (STAGE / "SHA256SUMS.txt").read_text() == expected
        print("release/ matches docs/SHA256SUMS.txt" if ok else "release/ differs")
        return 0 if ok else 1
    paths = yaml.safe_load((REPO / "configs/paths.local.yaml").read_text())
    rows = stage(cfg, Path(paths["outputs"]["runs"]))
    shutil.copy2(STAGE / "SHA256SUMS.txt", sums)
    (REPO / "docs/RELEASE.md").write_text(release_md(cfg, rows))
    print(f"staged {len(rows) + 1} files in release/; wrote docs/RELEASE.md, "
          "docs/SHA256SUMS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
