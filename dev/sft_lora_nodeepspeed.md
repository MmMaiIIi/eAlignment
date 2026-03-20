# SFT LoRA Fallback Without DeepSpeed

## Why This Fallback Exists

Current observed behavior:

- `sft_plain` (full fine-tuning, no LoRA) can hit CUDA OOM on `Qwen/Qwen3-8B`.
- `sft` (LoRA + DeepSpeed ZeRO-2) can fail at step 0 with an optimizer/scheduler param-group mismatch (`ValueError: zip() argument 2 is longer than argument 1`).

This fallback keeps LoRA memory efficiency but removes DeepSpeed to avoid the known scheduler integration failure path.

## New Profiles

- `sft_lora_nodeepspeed` -> `configs/sft_lora_nodeepspeed.yaml`
- `sft_lora_nodeepspeed_smoke` -> `configs/sft_lora_nodeepspeed_smoke.yaml`

Both are:

- `finetuning_type: lora`
- no `deepspeed` field
- single-process training path (no forced torchrun from DeepSpeed detection)

## Commands

Quick smoke check:

```bash
bash scripts/launch_sft.sh sft_lora_nodeepspeed_smoke
```

Fallback baseline run:

```bash
bash scripts/launch_sft.sh sft_lora_nodeepspeed
```

Through standard data-prep entrypoint:

```bash
bash runs/sft.sh sft_lora_nodeepspeed_smoke
bash runs/sft.sh sft_lora_nodeepspeed
```

## What Stays The Same

The SFT artifact pipeline stays unchanged and still writes:

- `runs/artifacts/sft/<run_name>/launch_command.txt`
- `runs/artifacts/sft/<run_name>/config_snapshot.yaml`
- `runs/artifacts/sft/<run_name>/run_manifest.json`
- `runs/artifacts/sft/<run_name>/training_log.jsonl`
- `runs/artifacts/sft/<run_name>/trainer_state.json` (if emitted)
- `runs/artifacts/sft/<run_name>/checkpoint_dirs.txt`
- `runs/artifacts/sft/<run_name>/tensorboard_event_files.txt`
- `runs/artifacts/sft/<run_name>/before_after.jsonl` (or `.md` fallback)
- `reports/loss_curve_<run_name>.png`
- `reports/loss_curve_<run_name>.json`
- `reports/sft_run_summary_<run_name>.md`

Before/after prompts are still sourced from:

- `data/synthetic/sft_eval_prompts.jsonl`

## Memory and Stability Tradeoffs

- More stable than LoRA + DeepSpeed in the known failing environment because DeepSpeed/ZeRO is fully removed from this profile.
- More memory-friendly than full fine-tuning because LoRA is retained.
- May still OOM on low-memory GPUs depending on sequence length and runtime fragmentation.
- Uses `fp16: true` for broad single-GPU compatibility. If your runtime is BF16-native and stable, you can switch this profile to BF16 later.
