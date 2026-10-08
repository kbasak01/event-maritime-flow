---
name: gpu-job
description: "Prepare a long GPU job (v2e conversion, training, evaluation, export benchmark) as a reproducible script plus manifest for the user to run, instead of running it. Use whenever a job would take more than about 2 minutes of GPU time."
argument-hint: "[job-name] [args...]"
---

# Prepare GPU job: $ARGUMENTS

Long jobs are run by the user. Your job is to make them one command, resumable and auditable.

1. Identify the plan phase and step this job serves; quote it.
2. Write `jobs/<date>_<job-name>.sh` (bash, `set -euo pipefail`) that:
   - activates the right conda env (`evflow`, or `v2e` for conversions);
   - sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the seed; disables TF32 for parity/export jobs;
   - runs each unit (one sequence / one arm-seed / one clip) as a separate command that **skips if
     its output and manifest already exist** (resumable);
   - writes logs to `runs/<job-name>/<unit>.log`;
   - ends by running the matching summariser so `results/*.json` is produced by a script.
3. Write `jobs/<date>_<job-name>.manifest.json`: plan step, git SHA, config files with SHA-256,
   inputs (split manifest SHA), expected outputs, estimated GPU-hours (from
   `results/v2e_throughput.json` or prior runs), disk estimate.
4. Run a dry check yourself: `bash -n` on the script and, if available, a `--smoke` / `--stop_time 5`
   variant of the first unit (must finish in < 2 min).
5. Print for the user, and stop:
   - the exact command (`nohup bash jobs/... > runs/<job-name>/job.log 2>&1 &` or a `tmux` line);
   - estimated duration and disk use;
   - what to tell you when it finishes ("job <name> finished").
6. Add the job to "Pending GPU jobs" in `docs/STATUS.md`.

Never start the full job yourself. Never point a training or calibration job at a test split.
