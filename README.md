# eAlignment: Interview-Ready LLM Alignment Project

This repository contains a framework-first alignment project for Chinese e-commerce customer support.

It is built around mature open-source frameworks instead of reimplementing training logic:

- Base model: `Qwen/Qwen3-8B`
- Primary training entrypoint: `LLaMA-Factory`
- Core stack: `transformers`, `trl`, `peft`, `accelerate`, `deepspeed`

## Why LLaMA-Factory

The project focuses on practical adaptation work:

- data preparation and schema validation
- configuration-driven SFT/DPO experiments
- evaluation and badcase analysis
- reproducible scripts and reports

Custom trainer, custom LoRA injection, and custom DPO loss code are intentionally out of scope.

## Repository Structure

```text
.
+-- AGENTS.md
+-- README.md
+-- requirements.txt
+-- pyproject.toml
+-- configs/
|   +-- llamafactory/
|   |   +-- sft/
|   |   +-- dpo/
|   |   +-- deepspeed/
|   +-- data/
|   +-- eval/
+-- data/
|   +-- raw/
|   +-- interim/
|   +-- processed/
|   +-- synthetic/
+-- scripts/
+-- src/
|   +-- data/
|   +-- eval/
|   +-- prompts/
|   +-- utils/
+-- reports/
|   +-- templates/
|   +-- experiments/
|   +-- badcases/
+-- tests/
```

## Stage Roadmap

- Stage 0: repository scaffold, dependencies, synthetic seeds, smoke checks
- Stage 1: schema validation and normalization/conversion scripts
- Stage 2: SFT experiment configs and launch wrappers
- Stage 3: evaluation pipeline, comparison flow, badcase collection
- Stage 4: preference pipeline and DPO configs
- Stage 5: ablations, final reports, interview packaging

## Setup

1. Create a Python environment (`3.10+` recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install `LLaMA-Factory` separately and ensure `llamafactory-cli` is available in your shell.
4. This repository does not vendor LLaMA-Factory internals; it only provides project-specific configs, scripts, and data adapters.

## Data Preparation

Convert raw/mock customer-support records into validated and normalized
LLaMA-Factory-ready datasets:

```bash
python scripts/prepare_sft_data.py
python scripts/prepare_dpo_data.py
```

Generated files:

- `data/processed/sft_all.jsonl`
- `data/processed/sft_train.jsonl`
- `data/processed/sft_dev.jsonl`
- `data/processed/sft_test.jsonl`
- `data/processed/dpo_all.jsonl`
- `data/processed/dpo_train.jsonl`
- `data/processed/dpo_dev.jsonl`
- `data/processed/dpo_test.jsonl`
- `data/processed/dataset_info.json`
- `data/interim/sft_rejected.jsonl`
- `data/interim/dpo_rejected.jsonl`

Raw/mock inputs for smoke runs:

- `data/raw/mock_sft_raw.jsonl`
- `data/raw/mock_dpo_raw.jsonl`

The Stage 1 pipeline is strict:

- normalizes category aliases to canonical values
- validates required schema fields
- rejects malformed records with explicit error reasons
- preserves metadata (`id`, `category`, `source`, `source_id`)
- splits valid records into train/dev/test using `configs/data/split.yaml`

## SFT Workflow

```bash
bash scripts/launch_sft.sh configs/llamafactory/sft/qwen3_8b_lora_sft.yaml
```

## DPO Workflow

```bash
bash scripts/launch_dpo.sh configs/llamafactory/dpo/qwen3_8b_lora_dpo.yaml
```

## Evaluation Workflow

Run proxy-rule evaluation and write per-sample scores:

```bash
python scripts/run_eval.py --predictions data/synthetic/eval_predictions.jsonl
python scripts/summarize_badcases.py --eval-file reports/experiments/latest_eval.jsonl
```

Notes:

- Current evaluation is proxy-based and rule-based (not human preference gold scoring).
- Proxy metrics are explicitly marked as proxies.

## Experiment Outputs

Expected output locations:

- training outputs: `outputs/`
- evaluation results: `reports/experiments/`
- badcase summaries: `reports/badcases/`

## Current Stage Status

- Stage 0 scaffold: completed
- Stage 1 data layer: completed (schema, validation, normalization, conversion, split)
- Stage 2+ implementation: incremental TODO

## Limitations

- No full training run is executed in this scaffold.
- Evaluation currently uses lightweight proxy checks.
- Mock/raw data is small and intended for smoke tests only.

## Interview Talking Points

- Why framework-first integration is more practical than rebuilding training internals
- How SFT/DPO data schemas are normalized for reproducibility
- How config-driven experimentation improves traceability
- How proxy evaluation and badcases guide iteration before costly training
