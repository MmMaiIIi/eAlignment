# AGENTS.md

## Architectural Intent

This repository is a compact adaptation layer around existing open-source training tooling for Chinese e-commerce customer-support alignment.

- Training backend: **LLaMA-Factory**
- Base model target: **Qwen/Qwen3-8B**
- This repo owns data prep, evaluation, badcases, ablation/reporting.
- This repo does **not** reimplement SFT/DPO trainers.

Main execution path:

`runs/ -> scripts/ -> align/`

## Directory Responsibilities

- `align/`
  Project core logic only: data normalization/validation, eval scoring/comparison, config/io helpers.
- `scripts/`
  Thin entrypoints for reproducible actions (`prepare_data`, `prepare_pref`, `eval`, `badcases`, `plan_ablations`).
- `runs/`
  Copy-paste runnable workflows (`smoke.sh`, `sft.sh`, `dpo.sh`, `eval.sh`).
- `configs/`
  Small config surface:
  - profile control (`profiles.yaml`)
  - LLaMA-Factory configs (`sft*.yaml`, `dpo*.yaml`, `ds_zero2.json`)
  - eval/ablation configs (`eval.yaml`, `ablations.yaml`)
- `data/`
  Raw, synthetic, processed, interim artifacts.
- `reports/`
  Templates, experiment outputs, badcase notes.
- `dev/`
  One-off temporary scripts only; not part of core runtime path.
- `tests/`
  Critical smoke checks for config/data/eval path.

## Core vs Script vs Runs vs Dev

- Put reusable logic in `align/`.
- Put CLI argument handling + path wiring in `scripts/`.
- Put short end-to-end command sequences in `runs/`.
- Put temporary or throwaway work in `dev/` only.

## What Must Never Be Reintroduced

1. Custom trainer layers duplicating LLaMA-Factory.
2. Giant `utils` dumping ground.
3. Unnecessary factories/registries/manager hierarchies.
4. Duplicate config universes that mirror upstream.
5. Dead wrappers with one caller and no value.
6. Hidden control flow that obscures the runtime path.

## Feature Addition Rules

1. Start from existing scripts and keep the execution path short.
2. Add new files only when an existing file would become unreadable.
3. Keep profile-based defaults; avoid adding knobs without clear value.
4. For training behavior changes, edit LLaMA-Factory configs directly.
5. For domain behavior changes, edit `align/data.py` or `align/eval.py`.

## New File Rules

- Each new top-level file/dir must have a clear operational reason.
- Avoid creating parallel modules for a single feature.
- Prefer editing an existing focused file over creating a tiny new one.

## Refactor Discipline

- Delete dead code promptly.
- Remove stale config keys and old paths when replacing workflows.
- Keep naming consistent with the main path.
- Do not preserve legacy indirection for sentiment.

## Testing Expectations

Minimum checks before finalizing meaningful changes:

1. Config loading + required field checks.
2. Data preparation smoke checks.
3. Evaluation pipeline smoke checks.

Use narrow test commands first; expand only when needed.

## README Discipline

When architecture or execution paths change, update README in the same change set:

- quickstart commands
- where to edit core logic
- verified vs expected workflows
- artifact paths
- limitations

