# Ablation Plan

- name: stage5_ablation_matrix

## Experiment 1: Base vs SFT (LoRA)
- id: base_vs_sft
- objective: Measure alignment gain from supervised fine-tuning.
- stage: sft
- train_config: configs/sft.yaml
- train_command: llamafactory-cli train configs/sft.yaml
- eval_command: python scripts/eval.py --config configs/eval.yaml --base data/synthetic/eval_base_predictions.jsonl --tuned data/synthetic/eval_sft_predictions.jsonl
- notes: 
- missing_paths: none

## Experiment 2: SFT vs SFT+DPO
- id: sft_vs_sft_dpo
- objective: Measure incremental gain after preference optimization.
- stage: dpo
- train_config: configs/dpo.yaml
- train_command: llamafactory-cli train configs/dpo.yaml
- eval_command: python scripts/eval.py --config configs/eval.yaml --base data/synthetic/eval_sft_predictions.jsonl --tuned data/synthetic/eval_sft_predictions.jsonl
- notes: 
- missing_paths: none

## Experiment 3: LoRA vs QLoRA
- id: lora_vs_qlora
- objective: Compare quality-efficiency tradeoff between LoRA and QLoRA.
- stage: sft
- train_config: configs/sft_qlora.yaml
- train_command: llamafactory-cli train configs/sft_qlora.yaml
- eval_command: python scripts/eval.py --config configs/eval.yaml --base data/synthetic/eval_base_predictions.jsonl --tuned data/synthetic/eval_sft_predictions.jsonl
- notes: 
- missing_paths: none

## Experiment 4: DPO beta sensitivity
- id: dpo_beta_comparison
- objective: Compare preference sharpness under different beta settings.
- stage: dpo
- train_config: configs/dpo.yaml
- train_command: llamafactory-cli train configs/dpo.yaml
- eval_command: python scripts/eval.py --config configs/eval.yaml
- notes: Adjust pref_beta, keep other settings fixed, rerun comparison.
- missing_paths: none

## Experiment 5: Preference-data quality sensitivity
- id: preference_quality_sensitivity
- objective: Compare strict vs relaxed preference filtering impacts.
- stage: dpo
- train_config: configs/dpo.yaml
- train_command: llamafactory-cli train configs/dpo.yaml
- eval_command: python scripts/eval.py --config configs/eval.yaml
- notes: Change filtering strictness in prepare_pref.py and rerun.
- missing_paths: none
