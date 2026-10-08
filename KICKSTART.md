# KICKSTART — running `event-maritime-flow` in Claude Code

## 0. One-time setup (you, in a terminal)

```bash
mkdir event-maritime-flow && cd event-maritime-flow && git init
unzip ~/Downloads/event-maritime-flow-kit.zip -d .        # or copy the kit files here
chmod +x .claude/hooks/guard_paths.sh                     # needs jq
cp configs/paths.local.yaml.example configs/paths.local.yaml   # fill in dataset paths
git add -A && git commit -m "Starter kit: plan, Claude Code agents and skills"
claude                                                    # accept the workspace-trust prompt
```

Check the kit loaded: ask "What skills and agents are available in this project?" — you should see the
5 skills and 8 agents listed in `CLAUDE.md`.

## 1. How the pieces are used

| You want to… | Type in Claude Code |
|---|---|
| close a phase | `/phase-gate 3` |
| get a long GPU job prepared (v2e conversion, training) | `/gpu-job v2e-mvsec` or `/gpu-job train v2e_rand_sup 2026` |
| convert a video/frame folder with v2e | `/v2e-convert smd MVI_0797_VIS_OB mvsec_cal` |
| check prose before committing it | `/claim-trace README.md` |
| force a specific auditor | `@agent-flow-metric-verifier check src/evflow/flow/` (or pick it from the `@` typeahead) |
| let Claude choose | just describe the task; agent descriptions route it |

Auditors are read-only and return a verdict; the `phase-gate` skill writes verdicts to
`docs/gates/phase-N.md`. `event-conventions` is reference knowledge Claude loads by itself.

## 2. Session-start prompt (paste at the start of every session)

```
Read CLAUDE.md, docs/STATUS.md and the current phase section of IMPLEMENTATION_PLAN.md.
Tell me in 5 lines: current phase, what is done, what is next, any open gate findings,
and which GPU jobs (if any) I need to run. Do not start editing until I confirm.
```

## 3. Phase prompts

Paste one phase prompt at a time. Each ends with the gate, so Claude stops at the boundary. A skill
name written mid-message gives Claude permission to run it; if Claude finishes without running the
gate, type `/phase-gate N` yourself at the start of a new message.

### Phase 0
```
Phase 0 of IMPLEMENTATION_PLAN.md. Create the package skeleton, pyproject.toml, Makefile, .gitignore,
CI workflow and docs/{STATUS,DECISIONS,LICENSES,LIMITATIONS}.md exactly as REPO_SETUP.md describes.
Write scripts/check_data.py (reads configs/paths.local.yaml, writes results/data_inventory.json).
Add tests/test_smoke.py, tests/test_no_gpl_imports.py, tests/test_results_have_tiers.py.
Then fill docs/LICENSES.md from SOURCES.md, listing every (verify) item as an open question for me —
do not guess licence terms. Print the environment-setup commands from REPO_SETUP.md §2 for me to run.
Finish with /phase-gate 0.
```

### Phase 1 (the crux — go slowly)
```
Phase 1. Load the event-conventions skill first. Write tests before implementation for:
io/store.py (canonical store + EventSlicer with DSEC ms_to_idx semantics, including the DSEC
docstring example), events/representations{,_ref}.py, flow/metrics{,_ref}.py, flow/iwe{,_ref}.py,
flow/fwl.py, flow/png.py. Build the analytic moving-edge fixture generator in tests/fixtures/.
Vendor the MIT metric functions from tudelft/event_flow into tests/oracles/ with their licence header.
Then the MVSEC and DSEC converters: before writing each, inspect one real file (h5ls -v / numpy) and
record the observed layout in docs/DECISIONS.md. Run make test, then make test-data on one MVSEC and
one DSEC file. Then run the flow-metric-verifier agent and the event-data-auditor agent. Finish with /phase-gate 1.
```

### Phase 2
```
Phase 2. Implement sim/v2e_runner.py, sim/stats.py and sim/calibrate.py per §1.7 and Phase 2 of the
plan, and configs/v2e_profiles.yaml. Use the v2e-convert skill for argument policy. First run a
--stop_time 5 trial yourself on 5 s of MVSEC APS (this is short enough). Then use /gpu-job to prepare
the throughput sweep and the calibration grid for me to run. When I report they finished, read the
JSON, freeze the profile and CT range, and log the corpus cuts in docs/DECISIONS.md.
Run the v2e-fidelity-auditor agent, then /phase-gate 2.
```

### Phase 3
```
Phase 3. Prepare /gpu-job v2e-mvsec for train/tune/test with mvsec_cal plus the K=4 randomised train
corpora. While it runs, write scripts/fidelity_report.py and the count-image cross-correlation check.
When I confirm the job finished, run the fidelity report and the alignment check, then
the v2e-fidelity-auditor agent and /phase-gate 3.
```

### Phase 4
```
Phase 4. Implement models/cm_baseline.py (M0b) and the zero-flow baseline. Vendor E-RAFT's model code
(MIT, licence header) into models/eraft_ref/ for inference only. Prepare /gpu-job for M0b and R1 on
mvsec_test and dsec_test, and the reproduction gate on E-RAFT's preprocessed outdoor_day1 20 Hz.
After results land: the flow-metric-verifier agent on the reproduction gate, then /phase-gate 4.
```

### Phase 5
```
Phase 5. Implement models/evflownet.py, losses/supervised.py, losses/cm.py (with an event-collapse
guard), train/loop.py and configs/train_<arm>.yaml for every MVSEC arm in §1.4. Make train-smoke pass.
Then /gpu-job each run (arms × seeds) and the representation ablation, as a single ordered script.
As runs finish, run the training-run-diagnostician agent on each. Finish with /phase-gate 5.
```

### Phase 6
```
Phase 6. Implement eval/run.py, eval/bootstrap.py, scripts/run_eval.py and scripts/make_tables.py
(Tables A–D, tier-tagged). Prepare /gpu-job eval-all. When results exist, run make_tables and
make_figures, then the sim2real-study-runner agent and the eval-reproducibility-auditor agent. Finish with /phase-gate 6.
```

### Phase 7
```
Phase 7. Implement io/smd.py, io/lars.py, teacher/raft.py and maritime/clutter.py. Inspect LaRS
image_annotations.json before choosing attributes. Prepare /gpu-job v2e-maritime, teacher-flows,
train-maritime and eval-maritime. Keep every T3 table in its own section. Then the sim2real-study-runner agent,
/claim-trace on any prose you wrote, and /phase-gate 7.
```

### Phase 8
```
Phase 8. Implement export/onnx.py and export/bench.py (P1 methodology). Prepare /gpu-job export-bench.
Then the onnx-parity-auditor agent and /phase-gate 8.
```

### Phase 9
```
Phase 9. Walk VALIDATION_CHECKLIST.md line by line; write verdict + evidence to docs/LIMITATIONS.md.
Run make reproduce-tables, the eval-reproducibility-auditor agent and a dry run of the repo-publication-reviewer agent.
List every unmet critical line. /phase-gate 9.
```

### Phase 10
```
Phase 10. Draft README.md per §Phase 10 of the plan. Run /claim-trace README.md, fix every finding,
then the repo-publication-reviewer agent. Prepare the release-asset list and SHA256SUMS.txt. /phase-gate 10.
```

## 4. Handy one-liners

- "Which tier is this number?" → `@agent-sim2real-study-runner tier-check results/eval/`
- "Is this run collapsed?" → `@agent-training-run-diagnostician runs/<run_id>`
- Resume after a break → the session-start prompt in §2.
- If Claude drifts from the plan: "Stop. Quote the plan step you are implementing and the exit gate."

## 5. When to deviate

If a gate fails twice, don't loosen it: ask Claude to write the failure, the evidence and two options
into `docs/DECISIONS.md`, and decide yourself. Typical legitimate cuts (log them): fewer MVSEC train
minutes after the Phase 2 throughput result; one seed for non-core arms; dropping DSEC to the fast path.
