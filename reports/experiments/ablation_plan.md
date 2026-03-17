# Ablation Plan

- name: stage5_ablation_matrix

## Experiment 1: Base vs SFT (LoRA)
- id: base_vs_sft
- objective: Measure alignment gain from supervised fine-tuning.
- train_stage: sft
- train_config: configs/llamafactory/sft/qwen3_8b_lora.yaml
- train_command: llamafactory-cli train configs/llamafactory/sft/qwen3_8b_lora.yaml
- eval_command: python scripts/run_eval.py --config configs/eval/comparison_eval.yaml --base-predictions data/synthetic/eval_base_predictions.jsonl --sft-predictions data/synthetic/eval_sft_predictions.jsonl
- notes: 
- missing_paths: none

## Experiment 2: SFT vs SFT+DPO
- id: sft_vs_sft_dpo
- objective: Measure additional preference alignment gain after DPO.
- train_stage: dpo
- train_config: configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml
- train_command: llamafactory-cli train configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml
- eval_command: python scripts/run_eval.py --config configs/eval/comparison_eval.yaml --base-predictions data/synthetic/eval_sft_predictions.jsonl --sft-predictions data/synthetic/eval_sft_predictions.jsonl
- notes: 
- missing_paths: none

## Experiment 3: LoRA vs QLoRA
- id: lora_vs_qlora
- objective: Compare quality-efficiency tradeoff between LoRA and QLoRA.
- train_stage: sft
- train_config: configs/llamafactory/sft/qwen3_8b_qlora.yaml
- train_command: llamafactory-cli train configs/llamafactory/sft/qwen3_8b_qlora.yaml
- eval_command: python scripts/run_eval.py --config configs/eval/comparison_eval.yaml --base-predictions data/synthetic/eval_base_predictions.jsonl --sft-predictions data/synthetic/eval_sft_predictions.jsonl
- notes: 
- missing_paths: none

## Experiment 4: DPO beta sensitivity
- id: dpo_beta_comparison
- objective: Compare preference sharpness at different beta values.
- train_stage: dpo
- train_config: configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml
- train_command: llamafactory-cli train configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml
- eval_command: python scripts/run_eval.py --config configs/eval/comparison_eval.yaml
- notes: Adjust pref_beta and rerun with consistent seeds.
- missing_paths: none

## Experiment 5: Preference-data quality sensitivity
- id: preference_quality_sensitivity
- objective: Measure impact of strict vs relaxed filtering on DPO outcomes.
- train_stage: dpo
- train_config: configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml
- train_command: llamafactory-cli train configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml
- eval_command: python scripts/run_eval.py --config configs/eval/comparison_eval.yaml
- notes: Compare runs with and without invalid/duplicate filtering.
- missing_paths: none
