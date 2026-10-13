"""Generated blocks of ``README.md`` (plan Phase 10; CLAUDE.md rule 3; D-067).

Every number in the README sits between ``<!-- BEGIN generated:NAME -->`` and
``<!-- END generated:NAME -->``; ``scripts/make_tables.py`` rewrites those blocks and its
``--check`` fails on a stale or hand-edited block (CI, ``make reproduce-tables``). Prose outside
the blocks carries no metric value.

Tables are cut verbatim from the rendered ``results/TABLES.md`` text (same records, same
rounding, same tier tags and footnotes); ``key-findings`` is templated from the committed JSON
with the qualifier of each sentence (sign stable across seeds, CI includes 0, ...) computed from
the numbers it quotes. ``eval-protocol`` quotes ``configs/eval_mvsec.yaml`` (checklist C4).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BEGIN = "<!-- BEGIN generated:{name} -->"
END = "<!-- END generated:{name} -->"
_MARK = re.compile(r"<!-- (BEGIN|END) generated:([a-z0-9-]+) -->")
BLOCKS = ("key-findings", "table-b-notes", "table-b-t1", "stats-note", "table-a-t1",
          "table-b-t2", "table-c", "table-d", "t3-protocol", "table-f-t3a", "table-g-t3b",
          "table-h-clutter", "table-i-parity", "table-j-latency", "eval-protocol")


class BlockError(ValueError):
    pass


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace the body of every marker pair in ``text``. Every block in the README needs a
    generator and every generated block must appear exactly once."""
    marks = [(m.group(1), m.group(2), m.start(), m.end()) for m in _MARK.finditer(text)]
    pairs, open_ = [], None
    for kind, name, s, e in marks:
        if kind == "BEGIN" and open_ is None:
            open_ = (name, e)
        elif kind == "END" and open_ is not None and open_[0] == name:
            pairs.append((name, open_[1], s))
            open_ = None
        else:
            raise BlockError(f"unbalanced marker {kind} {name}")
    if open_ is not None:
        raise BlockError(f"unbalanced marker BEGIN {open_[0]}")
    names = [p[0] for p in pairs]
    for n in set(names):
        if names.count(n) > 1:
            raise BlockError(f"block {n} appears more than once")
        if n not in blocks:
            raise BlockError(f"block {n} has no generator")
    for n in blocks:
        if n not in names:
            raise BlockError(f"block {n} is not in the README")
    out, pos = [], 0
    for name, s, e in pairs:
        out += [text[pos:s], "\n", blocks[name].strip("\n"), "\n"]
        pos = e
    out.append(text[pos:])
    return "".join(out)


def _level(line: str) -> int:
    m = re.match(r"^(#+) ", line)
    return len(m.group(1)) if m else 0


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def section(md: str, heading: str) -> list[str]:
    """Body of the first heading line starting with ``heading`` (up to the next heading of the
    same or a higher level), blank lines trimmed."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(heading) and _level(ln):
            lv, body = _level(ln), []
            for x in lines[i + 1:]:
                if _level(x) and _level(x) <= lv:
                    break
                body.append(x)
            return _trim(body)
    raise BlockError(f"heading {heading!r} not found in results/TABLES.md")


def chunk(lines: list[str], start: str) -> list[str]:
    """From the line starting with ``start`` through the end of the first table after it."""
    for i, ln in enumerate(lines):
        if ln.startswith(start):
            out, in_table = [], False
            for x in lines[i:]:
                if x.startswith("|"):
                    in_table = True
                elif in_table:
                    break
                out.append(x)
            return _trim(out)
    raise BlockError(f"{start!r} not found")


def _upto(lines: list[str], stop: str) -> list[str]:
    for i, ln in enumerate(lines):
        if ln.startswith(stop):
            return _trim(lines[:i])
    raise BlockError(f"{stop!r} not found")


def _from(lines: list[str], start: str) -> list[str]:
    for i, ln in enumerate(lines):
        if ln.startswith(start):
            return _trim(lines[i:])
    raise BlockError(f"{start!r} not found")


def _demote(lines: list[str], by: int = 1) -> list[str]:
    return [("#" * by + x) if _level(x) else x for x in lines]


# --- key findings ------------------------------------------------------------------------------

def _pf(v: float) -> str:
    """Same rounding as the Table B cells (make_tables._pf)."""
    return f"{v:.4f}" if v != 0 and abs(v) < 0.0005 else f"{v:.3f}"


def _px(e: dict) -> str:
    lo, hi = e["ci95"]
    return f"{_pf(e['estimate'])} px [95 % CI {_pf(lo)}, {_pf(hi)}]"


def _ci_word(e: dict) -> str:
    lo, hi = e["ci95"]
    return "excludes 0" if lo > 0 or hi < 0 else "includes 0"


def _seed_word(e: dict, n_pairs: str) -> str:
    r = (e.get("seed_pairings") or {}).get("range")
    if not r:
        return "one seed per arm"
    rng = f"[{_pf(r[0])}, {_pf(r[1])}]"
    if r[0] > 0:
        return f"positive in every one of the {n_pairs} single-seed pairings (range {rng})"
    if r[1] < 0:
        return f"negative in every one of the {n_pairs} single-seed pairings (range {rng})"
    return f"its sign changes across the {n_pairs} single-seed pairings (range {rng})"


def key_findings(root: Path) -> str:
    paired = json.loads((root / "results/eval/paired.json").read_text())
    P = {(e["table"], e["tier"], e["comparison"], e["sequence"], e.get("dt")): e
         for e in paired["entries"]}
    agg = json.loads((root / "results/eval/aggregate.json").read_text())
    A = {(e["tier"], e["arm"], e["dataset"], e["sequence"], e.get("dt")): e
         for e in agg["entries"] if e["representation"] == "voxel5"}
    zf = json.loads((root / "results/baselines/zero_flow/mvsec_outdoor_day1_dt1.json")
                    .read_text())
    n_seeds = len(A[("T1", "real_sup", "mvsec", "outdoor_day1", 1)]["seeds"])
    pairs = f"{n_seeds} × {n_seeds}"
    cpairs = f"{n_seeds} × {n_seeds} × {n_seeds}"
    od = "outdoor_day1"
    g1, g4 = P[("B", "T1", "gap_v2e_fixed_sup", od, 1)], P[("B", "T1", "gap_v2e_fixed_sup", od, 4)]
    r1 = P[("B", "T1", "gap_v2e_rand_sup", od, 1)]
    c1, c4 = (P[("B", "T1", "closure_v2e_rand_sup", od, 1)],
              P[("B", "T1", "closure_v2e_rand_sup", od, 4)])
    gp = P[("B", "T1", "gap_v2e_fixed_sup", "pooled", 1)]
    s_od, s_p = P[("B", "T1", "gap_selfsup", od, 1)], P[("B", "T1", "gap_selfsup", "pooled", 1)]
    t2 = P[("B", "T2", "gap_v2e_fixed_sup", od, 1)]
    rs = A[("T1", "real_sup", "mvsec", od, 1)]
    below = rs["ci95"]["epe"][1] < zf["ci95"]["epe"][0]

    def clo(e):
        if e["estimate"] is None:
            return "undefined (the fixed-CT gap's CI is not above 0)"
        return f"{_pf(e['estimate'])} [95 % CI {_pf(e['ci95'][0])}, {_pf(e['ci95'][1])}]"

    c = json.loads((root / "results/maritime/clutter.json").read_text())
    w = c["by_ct"]["ct100"]["pooled"]["water"]
    waves = c["attributes_ct100"]["waves"]["levels"]
    top = max(waves, key=lambda k: waves[k]["n"])
    par = json.loads((root / "results/onnx_parity.json").read_text())
    u = next(x for x in par["units"] if x["run_id"] == "real_sup_voxel5_s1337")
    fp32 = u["providers"]["ort_cuda_fp32"]["metrics"]
    lat = json.loads((root / "results/latency.json").read_text())
    e2e = next(x for x in lat["end_to_end"] if x["backend"] == "ort_cuda_fp32"
               and x["events"] == 100000 and x["width"] == 346)
    e2e_pt = next(x for x in lat["end_to_end"] if x["backend"] == "pytorch_eager_fp32"
                  and x["events"] == 100000 and x["width"] == 346)

    def ms(v):
        return f"{v:.3f}"

    L = [
        f"- **Matched-content simulator gap, outdoor_day1 (T1: real MVSEC test events, real "
        f"ground-truth flow; dt = 1 APS frame; {n_seeds} seeds per arm).** The same network "
        f"trained on calibrated v2e events vs on the real events of the same frames gives "
        f"EPE(`v2e_fixed_sup`) − EPE(`real_sup`) = {_px(g1)} (> 0: v2e training is worse); "
        f"the gap is {_seed_word(g1, pairs)}. At dt = 4 it is {_px(g4)}, "
        f"{_seed_word(g4, pairs)}. On this sequence at dt = 1, `real_sup` "
        + ("is clearly below the zero-flow EPE (CIs do not overlap)." if below else
           "is not clearly below the zero-flow EPE."),
        ("- **No measurable closure from contrast-threshold randomisation on outdoor_day1 "
         "(T1).** " if all(e["estimate"] is None or _ci_word(e) == "includes 0"
                           for e in (c1, c4)) else
         "- **Contrast-threshold randomisation on outdoor_day1 (T1).** ")
        + f"With {n_seeds} seeds per arm, `v2e_rand_sup` − `real_sup` = {_px(r1)} at dt = 1, "
        f"{_seed_word(r1, pairs)}; closure = {clo(c1)} at dt = 1 ({_seed_word(c1, cpairs)}) "
        f"and {clo(c4)} at dt = 4 ({_seed_word(c4, cpairs)}); closure 1 = gap fully closed, "
        "0 = no change.",
        f"- **Pooled over the four MVSEC test sequences (T1, dt = 1)** the fixed-CT gap is "
        f"{_px(gp)} and {_seed_word(gp, pairs)}: no stable sign. On the indoor sequences the "
        "supervised arms sit at or near the zero-flow floor (D-046), so pooled gaps mostly "
        "compare models near that floor, not a v2e advantage.",
        f"- **Self-supervised arms (T1, one seed each).** `v2e_selfsup` − `real_selfsup` = "
        f"{_px(s_p)} pooled at dt = 1 (CI {_ci_word(s_p)}) and {_px(s_od)} on outdoor_day1 at "
        f"dt = 1 (CI {_ci_word(s_od)}); single-seed results, not seed-tested, and the pooled value "
        "carries the same indoor zero-flow-floor caveat.",
        f"- **Scored on v2e(APS) test events instead (T2, outdoor_day1, dt = 1, {n_seeds} "
        f"seeds per arm)**, the same fixed-CT gap is {_px(t2)}, {_seed_word(t2, pairs)}"
        + (": on simulated test events the v2e-trained arm has the lower EPE, the opposite "
           "sign to T1, so a sim-only benchmark would hide the T1 gap." if t2["ci95"][1] < 0
           and g1["ci95"][0] > 0 else f" (CI {_ci_word(t2)})."),
        f"- **Sea clutter in synthetic v2e(LaRS) events (T3b, {c['n_clips']} `lars_eval` "
        f"keyframe windows, calibrated CT ×1, LaRS frame rate assumed 30 fps).** Water covers "
        f"{w['pixel_share']['mean']:.3f} of the pixels but receives "
        f"{w['event_share']['mean']:.3f} of the events (enrichment "
        f"{w['enrichment']['mean']:.3f} [95 % CI {w['enrichment']['ci95'][0]:.3f}, "
        f"{w['enrichment']['ci95'][1]:.3f}]). This describes v2e on these frames, not a real "
        f"sensor; {waves[top]['n']} of the {c['n_clips']} clips are labelled `{top}` waves.",
        f"- **Deployment (fp32 only, D-064).** The fp32 ONNX graph (ONNX Runtime CUDA EP) of "
        f"the `{u['arm']}` seed-{u['seed']} checkpoint (lowest `mvsec_tune` EPE, D-063) "
        f"matches PyTorch within {fp32['max_px']:.2e} px max |Δflow| on "
        f"{u['n_windows']} real `mvsec_test` windows (T1). End-to-end events → flow at 346×260 "
        f"with 10⁵ events per window: p50 {ms(e2e['total']['p50_ms'])} ms (ORT CUDA fp32, "
        f"composed) and {ms(e2e_pt['total']['p50_ms'])} ms (PyTorch eager fp32, measured) on "
        f"an RTX A4000 that stands in for an embedded target (same checkpoint, "
        f"`{lat['model']['arm_run']}`). Hardware timing of the network, "
        "not event-sensor latency; the fp16 graphs failed the parity tolerance.",
    ]
    return "\n".join(L)


def eval_protocol(root: Path) -> str:
    lines = (root / "configs/eval_mvsec.yaml").read_text().splitlines()
    keep = []
    for ln in lines:
        if ln.startswith("reproduction_eraft:"):
            break
        keep.append(ln)
    return "\n".join(["```yaml", *_trim(keep), "```"])


def render_blocks(root: Path, tables_md: str) -> dict[str, str]:
    T = tables_md
    b = section(T, "## Table B")
    a1 = section(T, "## Table A-T1")
    j = section(T, "## Table J")
    out = {
        "key-findings": key_findings(root),
        "table-b-notes": "\n".join(_upto(b, "### Table B-T1")),
        "table-b-t1": "\n".join(section(T, "### Table B-T1")),
        "stats-note": "\n".join(chunk_paragraph(T, "**Statistics.**")),
        "table-a-t1": "\n".join(chunk(a1, "**EPE (px), dt = 1**") + [""]
                                + chunk(a1, "**EPE (px), dt = 4**") + [""]
                                + _from(a1, "Notes:")),
        "table-b-t2": "\n".join(section(T, "### Table B-T2")),
        "table-c": "\n".join(section(T, "## Table C")),
        "table-d": "\n".join(_upto(section(T, "## Table D"), "**EPE (px)**") + [""]
                             + chunk(section(T, "## Table D"), "**EPE (px)**")),
        "t3-protocol": "\n".join(section(T, "# Maritime tables (T3)")[:3]),
        "table-f-t3a": "\n".join(chunk(section(T, "## Table F-T3a"), "**teacher-EPE")),
        "table-g-t3b": "\n".join(_upto(section(T, "## Table G-T3b"), "**FWL") + [""]
                                 + chunk(section(T, "## Table G-T3b"), "**FWL")),
        "table-h-clutter": "\n".join(_upto(section(T, "## Table H"), "**Water enrichment")),
        "table-i-parity": "\n".join(_upto(section(T, "## Table I-T1"), "**fp16 sensitivity")),
        "table-j-latency": "\n".join(_demote(j, 1)),
        "eval-protocol": eval_protocol(root),
    }
    assert set(out) == set(BLOCKS)
    return out


def chunk_paragraph(md: str, start: str) -> list[str]:
    for ln in md.splitlines():
        if ln.startswith(start):
            return [ln]
    raise BlockError(f"{start!r} not found")
