---
name: phase-gate
description: "Close a phase of IMPLEMENTATION_PLAN.md by checking its exit gate, running the required tests and auditor agents, and recording the verdict. Use when the user types /phase-gate N or asks to close or check a phase."
argument-hint: "[phase-number]"
---

# Phase gate for phase $ARGUMENTS

1. Read the "Phase $ARGUMENTS" section and its **Exit gate** line in `IMPLEMENTATION_PLAN.md`, the
   matching `VALIDATION_CHECKLIST.md` lines, and `docs/STATUS.md`.
2. Run, and quote the tail of the output for each:
   - `make lint` and `make test` (always);
   - `make test-data` if the phase touched `io/`, `sim/` or `eval/` and data paths exist;
   - `make train-smoke` for phase 5; `make reproduce-tables` for phases 6, 7, 9, 10.
3. Invoke the required agents for this phase (one at a time, read-only):

   | Phase | Agents |
   |---|---|
   | 0 | repo-publication-reviewer (licence table only) |
   | 1 | flow-metric-verifier, event-data-auditor |
   | 2 | v2e-fidelity-auditor, event-data-auditor (splits) |
   | 3 | v2e-fidelity-auditor, event-data-auditor |
   | 4 | flow-metric-verifier |
   | 5 | training-run-diagnostician (per run), event-data-auditor (selection leakage) |
   | 6 | sim2real-study-runner, eval-reproducibility-auditor |
   | 7 | sim2real-study-runner, eval-reproducibility-auditor |
   | 8 | onnx-parity-auditor |
   | 9 | eval-reproducibility-auditor, repo-publication-reviewer (dry run) |
   | 10 | repo-publication-reviewer |

4. Check every artefact the exit gate names exists and is committed (`git ls-files`); check
   results JSON carry the keys required by the `event-conventions` skill.
5. Write `docs/gates/phase-$ARGUMENTS.md`:
   - gate text quoted from the plan;
   - each criterion → PASS / FAIL / N/A with evidence path or command output;
   - each agent's verdict verbatim (first line) and its critical findings;
   - open items and who must act (user / Claude).
6. Overall verdict: **PASS** only if every criterion passes and no agent returned a blocking verdict
   (FAIL, UNSOUND, NOT REPRODUCED, UNTRUSTWORTHY, BLOCK). Otherwise **FAIL**, with the shortest fix list.
7. Update the phase row in `docs/STATUS.md` and list pending GPU jobs. Do not start the next phase.

Never loosen a gate threshold to pass. If a threshold looks wrong, say so and propose a
`docs/DECISIONS.md` entry for the user to approve.
