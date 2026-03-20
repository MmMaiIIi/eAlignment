# Minimal SFT Logging Loop

## Purpose

This document explains the minimum logging bundle produced by one successful SFT run using the existing path:

`runs/sft.sh -> scripts/launch_sft.sh -> llamafactory-cli train ...`

No framework switch is required.

## Artifact Locations

Per-run machine-readable artifacts live under:

`runs/artifacts/sft/<run_name>/`

Rendered run summaries live under:

- `reports/loss_curve_<run_name>.png`
- `reports/loss_curve_<run_name>.json`
- `reports/sft_run_summary_<run_name>.md`

## Run Name

By default, run name is:

`<UTC timestamp>_<profile tag>_<model tag>`

Example:

`20260320T142530Z_sft_qwen3-8b`

You can override explicitly:

`SFT_RUN_NAME=my_custom_run bash scripts/launch_sft.sh sft`

## What A Successful Run Leaves Behind

Under `runs/artifacts/sft/<run_name>/`:

- `launch_command.txt`
- `config_snapshot.yaml`
- `run_manifest.json`
- `console.log`
- `training_log.jsonl`
- `trainer_state.json` (if emitted by trainer)
- `checkpoint_dirs.txt` (paths of checkpoint directories found)
- `tensorboard_event_files.txt` (paths of event files found)
- `before_after.jsonl` when generation comparison succeeds, otherwise `before_after.md`

Under `reports/`:

- `loss_curve_<run_name>.png`
- `loss_curve_<run_name>.json`
- `sft_run_summary_<run_name>.md`

## Fixed Prompt Set For Qualitative Comparison

Prompt source is fixed and deterministic:

`data/synthetic/sft_eval_prompts.jsonl`

The launcher uses this file in two phases:

1. Before training: generate base-model outputs and save `before_outputs.jsonl`
2. After training: generate tuned-model outputs and write `before_after.jsonl`

This keeps comparison reproducible and avoids ad hoc prompt copy/paste.

## How Loss Logging Works

After training, the launcher creates persistent machine-readable loss logs:

1. Preferred source: `<output_dir>/trainer_log.jsonl` if emitted by the framework
2. Fallback source: `<output_dir>/trainer_state.json` `log_history`
3. If neither exists, `training_log.jsonl` is still written with a warning record

Loss plotting command (run automatically at the end of launch):

`python scripts/plot_loss.py --run-name <run_name> --training-log runs/artifacts/sft/<run_name>/training_log.jsonl --trainer-state runs/artifacts/sft/<run_name>/trainer_state.json --output-png reports/loss_curve_<run_name>.png --output-json reports/loss_curve_<run_name>.json`

## Manifest

`run_manifest.json` contains pointers to:

- launch command and config snapshot
- dataset and output locations
- training log source
- trainer state path
- checkpoint pointers
- tensorboard event pointers
- loss curve report paths
- before/after comparison artifacts

## Environment Flags

- `SFT_RUN_NAME`: override run name
- `SFT_RUN_TIMESTAMP`: override timestamp segment
- `SFT_EVAL_PROMPTS_PATH`: override prompt file path
- `SFT_SKIP_BEFORE_AFTER=1`: skip generation comparison
- `SFT_COMPARE_MAX_NEW_TOKENS`: generation length cap (default `96`)
- `SFT_COMPARE_TEMPERATURE`: generation temperature (default `0.0`)
- `SFT_COMPARE_TOP_P`: top-p (default `1.0`)
- `SFT_COMPARE_REPETITION_PENALTY`: repetition penalty (default `1.0`)

## Known Limitations

- Checkpoints, trainer state, and tensorboard events are framework-emitted artifacts. If the underlying training runtime does not emit them in a given environment, pointer files will be empty.
- Before/after generation depends on local model loading support (`transformers`, `torch`, and adapter loading via `peft` when needed).
- Large base models increase run time for the before/after step even with a short fixed prompt list.
