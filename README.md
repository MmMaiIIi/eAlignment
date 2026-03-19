# ecommerce-support-alignment

Framework-first LLM alignment project for Chinese e-commerce customer support.

## What This Project Does

This repository provides a compact, reproducible workflow for:

- SFT data preparation
- preference data preparation for DPO
- LLaMA-Factory launch paths for SFT and DPO
- offline evaluation (base vs tuned)
- badcase collection
- ablation planning/report templates

Target domains:

- returns/refunds
- shipping/logistics
- product specs
- order modification
- after-sales
- complaint soothing

## Why LLaMA-Factory (Not Custom Trainer Code)

Training internals are delegated to **LLaMA-Factory** by design:

- no custom SFT trainer
- no custom DPO trainer/loss
- no custom LoRA/QLoRA internals
- no custom distributed runtime

This repo owns only project-specific adaptation logic.

## Architecture In One Paragraph

The repository follows a shallow path: `runs -> scripts -> align`.  
`align/` holds reusable data/eval logic, `scripts/` are explicit CLI entrypoints, and `runs/` are short reproducible command bundles. Config is compressed around a small **profile** knob with strong defaults.

## Main Execution Path

1. Prepare SFT data: `scripts/prepare_data.py`
2. Prepare preference data: `scripts/prepare_pref.py`
3. Launch SFT/DPO through LLaMA-Factory: `scripts/launch_sft.sh`, `scripts/launch_dpo.sh`
4. Run offline eval: `scripts/eval.py`
5. Summarize badcases: `scripts/badcases.py`

## Directory Tree

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── runs/
├── scripts/
├── align/
├── configs/
├── data/
├── reports/
├── dev/
└── tests/
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --profile smoke --source-format internal
```

Shortest working command:

```bash
bash runs/smoke.sh
```

## Setup Assumptions

- Python 3.10+
- `llamafactory-cli` installed separately and available in shell
- local environment has permissions to read/write `data/` and `reports/`

Verified in-repo:

- data prep/eval/badcase scripts run on mock/synthetic assets
- config and command-construction smoke tests

Expected (environment-dependent):

- full GPU SFT/DPO training runs
- full checkpoint export and large-scale eval

## Data Preparation

Pipeline:

`raw source rows -> source normalizer -> unified raw SFT schema -> validation/split -> processed jsonl + dataset_info -> LLaMA-Factory`

Supported SFT source formats:

- `internal`
- `jddc`
- `ecd`
- `faq`

Unified raw SFT schema (normalizer output):

```json
{
  "id": "...",
  "category": "...",
  "query": "...",
  "response": "...",
  "source": "...",
  "source_id": "...",
  "system": "optional",
  "metadata": "optional"
}
```

SFT:

```bash
python scripts/prepare_data.py --profile sft --input data/raw/merged_sft.jsonl --source-format internal --source-name merged
python scripts/prepare_data.py --profile sft --input data/raw/jddc_sft.jsonl --source-format jddc --source-name jddc
python scripts/prepare_data.py --profile sft --input data/raw/ecd_sft.jsonl --source-format ecd --source-name ecd
python scripts/prepare_data.py --profile sft --input data/raw/faq_sft.jsonl --source-format faq --source-name faq
```

Preference:

```bash
python scripts/prepare_pref.py --profile smoke
```

Outputs:

- `data/processed/sft_{all,train,dev,test}.jsonl`
- `data/processed/dpo_{all,train,dev,test}.jsonl`
- `data/processed/dataset_info.json`
- `data/interim/sft_rejected.jsonl`
- `data/interim/dpo_rejected.jsonl`
- `data/interim/dpo_quality_report.json`

Where to add a new SFT source:

- edit `align/data.py` in `_normalize_external_sft`
- keep it as `external -> unified raw schema -> existing validator`

## SFT

LLaMA-Factory takeover point:

- `scripts/launch_sft.sh` runs `llamafactory-cli train <configs/sft*.yaml>`
- `dataset: ecom_sft_seed` + `dataset_dir: data/processed` are resolved by `data/processed/dataset_info.json`

Dry run:

```bash
DRY_RUN=1 bash scripts/launch_sft.sh smoke
```

Actual launch (baseline LoRA):

```bash
bash scripts/launch_sft.sh sft
```

QLoRA variant:

```bash
bash scripts/launch_sft.sh sft_qlora
```

## DPO

Dry run:

```bash
DRY_RUN=1 bash scripts/launch_dpo.sh smoke
```

Actual launch:

```bash
bash scripts/launch_dpo.sh dpo
```

## Evaluation

```bash
python scripts/eval.py --profile eval
python scripts/badcases.py --profile eval --top-k 10
```

Outputs:

- `reports/experiments/latest_eval/summary.json`
- `reports/experiments/latest_eval/per_sample.jsonl`
- `reports/experiments/latest_eval/badcases.jsonl`
- `reports/experiments/latest_eval/report.md`
- `reports/badcases/latest_badcases.md`

## Artifacts

- Training output roots:
  - `outputs/sft/...`
  - `outputs/dpo/...`
- Eval artifacts: `reports/experiments/latest_eval/`
- Badcase notes/templates: `reports/badcases/`, `reports/templates/`
- Ablation plan artifacts:
  - `reports/experiments/ablation_plan.md`
  - `reports/experiments/ablation_plan.json`

## Ablation Planning

```bash
python scripts/plan_ablations.py --check-paths
```

Control file: `configs/ablations.yaml`

## Where To Modify Core Logic

- Data normalization/validation/splitting: `align/data.py`
- Evaluation scoring/comparison/badcase extraction: `align/eval.py`
- Profile/config loading: `align/config.py`

Temporary one-off work belongs in `dev/` only.

## Design Principles Used In This Rebuild

- framework-first boundary with LLaMA-Factory
- explicit scripts over hidden orchestration
- strong defaults over many knobs
- profile-based control (`smoke`, `sft`, `dpo`, `eval`)
- low abstraction and short execution path
- honest verified-vs-expected documentation

## Limitations

- Evaluation metrics are proxy heuristics, not benchmark-grade truth.
- Included datasets are mock/synthetic and small.
- No fabricated training results; real gains require actual runs.

## Interview Talking Points

- Why framework-first beats custom trainer reimplementation for project scope
- How preference data quality checks reduce DPO noise
- How side-by-side eval + badcases drive practical iteration
- How profile compression keeps experimentation fast and reproducible
