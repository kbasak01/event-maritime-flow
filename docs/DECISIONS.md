# DECISIONS

Format: `D-NNN · date · phase · decision · reason · evidence file`.

- D-001 · 2026-10-08 · plan · Standalone repo (no P1 submodule); reuse P1 latency methodology only. · P5 shares no model code with P1. · IMPLEMENTATION_PLAN.md §intro
- D-002 · 2026-10-08 · plan · LaRS + SMD replace MaSTr1325 as maritime video sources. · MaSTr1325 has no sequences; v2e needs consecutive frames. · SOURCES.md
- D-003 · 2026-10-08 · plan · MVSEC is the matched-content (T2) testbed; DSEC is T1 generalisation only. · DAVIS APS shares the DVS pixel grid; DSEC frame cameras are separate sensors. · IMPLEMENTATION_PLAN.md §0(3)
- D-004 · 2026-10-08 · plan · GPL-3.0 tub-rip code is never vendored/imported. · Repo is MIT. · SOURCES.md
