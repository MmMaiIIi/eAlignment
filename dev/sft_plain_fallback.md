# SFT Plain Fallback (No LoRA, No DeepSpeed)

## Why This Fallback Exists

This fallback provides a simpler SFT path for debugging and baseline verification when the LoRA + DeepSpeed path is unstable.

It intentionally disables:

- LoRA / PEFT adapter training
- DeepSpeed / ZeRO

So the runtime path is a plain single-process SFT launch through the existing flow:

`runs/sft.sh -> scripts/launch_sft.sh -> llamafactory-cli train ...`

## New Profiles

- `sft_plain` -> `configs/sft_plain.yaml`
- `sft_plain_smoke` -> `configs/sft_plain_smoke.yaml`

Both configs are full fine-tuning style (`finetuning_type: full`) with no `deepspeed` field.

## When To Use

Use this fallback when:

- LoRA + DeepSpeed fails in optimizer/scheduler integration
- You need a minimal training path to verify end-to-end artifacts
- You want a lower-complexity baseline for debugging

## Commands

Plain smoke (fast sanity run):

```bash
bash scripts/launch_sft.sh sft_plain_smoke
```

Plain full fallback run:

```bash
bash scripts/launch_sft.sh sft_plain
```

Using the standard run entrypoint with data prep:

```bash
bash runs/sft.sh sft_plain_smoke
bash runs/sft.sh sft_plain
```

## Artifact Expectations

A successful run still uses the same artifact workflow:

- `runs/artifacts/sft/<run_name>/launch_command.txt`
- `runs/artifacts/sft/<run_name>/config_snapshot.yaml`
- `runs/artifacts/sft/<run_name>/run_manifest.json`
- `runs/artifacts/sft/<run_name>/training_log.jsonl`
- `runs/artifacts/sft/<run_name>/trainer_state.json` (if emitted)
- `runs/artifacts/sft/<run_name>/checkpoint_dirs.txt`
- `runs/artifacts/sft/<run_name>/tensorboard_event_files.txt`
- `runs/artifacts/sft/<run_name>/before_after.jsonl` (or `before_after.md` fallback)
- `reports/loss_curve_<run_name>.png`
- `reports/loss_curve_<run_name>.json`
- `reports/sft_run_summary_<run_name>.md`

Prompt source for before/after remains:

- `data/synthetic/sft_eval_prompts.jsonl`

## Tradeoffs

- Simpler runtime and fewer moving parts
- Higher memory usage than LoRA/QLoRA
- Full fine-tuning on `Qwen/Qwen3-8B` may still require a large single GPU

If memory is insufficient, prefer `sft_plain_smoke` first, then reduce `cutoff_len`, `max_samples`, or run on higher-memory hardware.
