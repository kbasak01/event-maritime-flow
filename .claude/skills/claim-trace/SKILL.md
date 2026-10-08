---
name: claim-trace
description: "Trace every number and claim in a prose file (README, docs, figure captions) to committed results JSON, check evidence-tier labels and forbidden wording. Use before committing any README or docs text that states results."
argument-hint: "[file]"
---

# Claim trace for $ARGUMENTS

1. Extract every number (with its sentence) and every comparative/qualitative claim from
   `$ARGUMENTS`.
2. For each number: find the `results/**/*.json` field it came from (search by value and context).
   Record `file:key`. A number with no source, or that differs from its source after rounding,
   is a **finding**.
3. For each number: check the sentence or table states its tier (T1/T2/T3a/T3b), dataset, sequence,
   dt/window, arm and seed count. Missing ⇒ finding.
4. Wording checks (finding if present):
   - "HDR", "high dynamic range", "glare", "low latency", "microsecond" attached to v2e-derived results;
   - "EPE" or "accuracy" for T3a (must be "teacher-EPE"); "accuracy" for FWL/RSAT;
   - "ground truth" for RAFT teacher flow;
   - "first", "novel", "state of the art" without support from IMPLEMENTATION_PLAN.md §0 positioning
     or a results row;
   - "significant" without a CI or test; "guarantee".
5. Check every LaRS number states the frame-rate assumption, and every MVSEC number the eval window
   convention (or links to it).
6. Output a table: line · claim · source (file:key) · tier ok? · wording ok? · verdict. End with
   CLEAN or N findings. Do not edit the file unless asked; propose exact replacement text.
