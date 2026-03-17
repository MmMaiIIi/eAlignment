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

Example checks:

```bash
llamafactory-cli --help
python scripts/prepare_sft_data.py
```

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

Stage 2 keeps SFT config-driven through LLaMA-Factory.

Available SFT configs:

- `configs/llamafactory/sft/smoke.yaml`
- `configs/llamafactory/sft/qwen3_8b_lora.yaml`
- `configs/llamafactory/sft/qwen3_8b_qlora.yaml`

Launch examples:

```bash
# quick smoke run
bash scripts/launch_sft.sh configs/llamafactory/sft/smoke.yaml

# default LoRA run for Qwen3-8B
bash scripts/launch_sft.sh configs/llamafactory/sft/qwen3_8b_lora.yaml

# QLoRA run
bash scripts/launch_sft.sh configs/llamafactory/sft/qwen3_8b_qlora.yaml

# preview command without execution
DRY_RUN=1 bash scripts/launch_sft.sh configs/llamafactory/sft/smoke.yaml
```

Model export example:

```bash
DRY_RUN=1 bash scripts/export_model.sh configs/llamafactory/sft/qwen3_8b_lora.yaml
```

## DPO Workflow

Preference data format (normalized JSONL):

```json
{
  "id": "pref_0001",
  "category": "complaint_soothing",
  "prompt": "Customer: Your delivery is late again.",
  "chosen": "I am sorry for the delay. Please share your order number and I will check the latest status.",
  "rejected": "Wait patiently."
}
```

Preference quality checks in `scripts/prepare_dpo_data.py` include:

- empty `chosen` / `rejected`
- identical `chosen` and `rejected`
- malformed examples (schema/field issues)
- category imbalance analysis
- duplicate prompt/pair pattern analysis

Generated quality report:

- `data/interim/dpo_quality_report.json`

Available DPO configs:

- `configs/llamafactory/dpo/smoke.yaml`
- `configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml`

Launch examples:

```bash
# quick smoke run
bash scripts/launch_dpo.sh configs/llamafactory/dpo/smoke.yaml

# default LoRA DPO run for Qwen3-8B
bash scripts/launch_dpo.sh configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml

# preview command without execution
DRY_RUN=1 bash scripts/launch_dpo.sh configs/llamafactory/dpo/smoke.yaml
```

## Evaluation Workflow

Stage 3 evaluation supports side-by-side base vs SFT comparison using prepared prediction JSONL files.

Default synthetic comparison inputs:

- `data/synthetic/eval_base_predictions.jsonl`
- `data/synthetic/eval_sft_predictions.jsonl`

`eval_sft_predictions.jsonl` can come from either:

- a full SFT checkpoint
- adapter-based inference output (LoRA/QLoRA)

Run evaluation:

```bash
python scripts/run_eval.py \
  --config configs/eval/comparison_eval.yaml \
  --base-predictions data/synthetic/eval_base_predictions.jsonl \
  --sft-predictions data/synthetic/eval_sft_predictions.jsonl \
  --output-dir reports/experiments/latest_eval

python scripts/summarize_badcases.py \
  --badcase-file reports/experiments/latest_eval/badcases.jsonl \
  --output-md reports/badcases/latest_badcases.md
```

Generated outputs:

- `reports/experiments/latest_eval/summary.json`
- `reports/experiments/latest_eval/per_sample.jsonl`
- `reports/experiments/latest_eval/badcases.jsonl`
- `reports/experiments/latest_eval/report.md`

Evaluation dimensions:

- actionability
- politeness/tone
- policy compliance
- low-quality/repetition risk
- category-level breakdown
- base vs SFT delta and win counts

Metric interpretation:

- exact metrics: response length, token count, repetition ratio
- proxy metrics: actionability/politeness/policy/quality rule-based heuristics
- proxy metrics are directional signals, not benchmark ground truth

## Experiment Outputs

Expected output locations:

- training outputs: `outputs/`
- SFT smoke artifacts: `outputs/sft/smoke/`
- SFT LoRA artifacts: `outputs/sft/qwen3_8b_lora/`
- SFT QLoRA artifacts: `outputs/sft/qwen3_8b_qlora/`
- DPO smoke artifacts: `outputs/dpo/smoke/`
- DPO LoRA artifacts: `outputs/dpo/qwen3_8b_dpo_lora/`
- evaluation results: `reports/experiments/`
- badcase summaries: `reports/badcases/`

## Current Stage Status

- Stage 0 scaffold: completed
- Stage 1 data layer: completed (schema, validation, normalization, conversion, split)
- Stage 2 SFT experiment layer: completed (LLaMA-Factory configs + wrappers + smoke checks)
- Stage 3 evaluation layer: completed (comparison pipeline + badcase collection + reports)
- Stage 4 preference + DPO layer: completed (quality checks + DPO configs + wrappers + smoke checks)
- Stage 5 implementation: incremental TODO

## Limitations

- No full training run is executed in this scaffold.
- Evaluation currently uses lightweight proxy checks.
- Mock/raw data is small and intended for smoke tests only.

## Interview Talking Points

- Why framework-first integration is more practical than rebuilding training internals
- How SFT/DPO data schemas are normalized for reproducibility
- How config-driven experimentation improves traceability
- How proxy evaluation and badcases guide iteration before costly training
