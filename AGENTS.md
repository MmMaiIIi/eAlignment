# AGENTS.md

## Project overview

This repository implements a staged alignment training pipeline for a domain-specific assistant.
Current default domain: e-commerce customer support.
Primary base model: Qwen3-8B-Instruct.
Primary training stages:
1. data schema and validation
2. supervised fine-tuning (SFT)
3. preference optimization with DPO
4. evaluation, badcase analysis, and reporting

The repo is intended for:
- reproducible local development
- smoke-testable training workflows
- interview-ready ML engineering demonstration
- incremental agent-driven implementation with Codex

## How Codex should work in this repo

Always work stage by stage.
Do not implement future stages unless explicitly asked.
Do not rewrite the whole repo when a local change is sufficient.
Prefer the smallest correct change set.
Before editing, inspect the current tree and read relevant files first.
After edits, run the required validation commands for the current stage.
If a command fails, fix the failure or clearly explain the blocker in the final summary.

For complex tasks, first produce a short implementation plan inside the work log, then execute.
Do not stop at planning unless the user explicitly asks for planning only.

## Non-goals

Do not introduce distributed multi-node training in early stages.
Do not add RLHF PPO, GRPO, ORPO, or reward-model training unless explicitly requested.
Do not add web UI, dashboards, or demo frontend unless explicitly requested.
Do not add extra datasets from the internet unless explicitly requested.
Do not optimize for benchmark SOTA. Optimize for clean, correct, inspectable engineering.

## Engineering principles

- Keep code simple and explicit.
- Prefer typed Python where practical.
- Prefer deterministic behavior in preprocessing and tests.
- Keep functions short and composable.
- Separate config, data, training, evaluation, and reports cleanly.
- Every training script must support a smoke-test mode.
- Every stage must leave the repo in a runnable state.

## Expected repository structure

- `README.md`                     project overview and usage
- `pyproject.toml` or `requirements.txt`
- `src/`
  - `config/`
  - `data/`
  - `training/`
  - `eval/`
  - `utils/`
- `scripts/`
- `configs/`
- `tests/`
- `data/`
  - `raw/`
  - `processed/`
  - `samples/`
- `artifacts/`
- `reports/`

If the actual repo structure differs, adapt carefully instead of forcing a full rewrite.

## Standard Python environment

Target Python: 3.10 or 3.11

Preferred core libraries:
- transformers
- datasets
- peft
- trl
- accelerate
- bitsandbytes if needed for QLoRA
- pydantic or dataclasses for schema validation
- pytest
- pandas
- scikit-learn for lightweight metrics if needed
- matplotlib or plotly only if charts are explicitly needed

Do not add large optional dependencies unless they are justified by a current stage requirement.

## Configuration rules

Use config files for anything that may change between runs:
- model name
- dataset paths
- output directories
- LoRA hyperparameters
- training hyperparameters
- eval hyperparameters
- random seeds

Prefer YAML or JSON for user-facing configs.
Do not hardcode machine-specific absolute paths.

## Data rules

All datasets must be versionable and inspectable.
Keep sample data in-repo for smoke tests.
If full data is unavailable, create realistic synthetic sample data that matches the intended schema.
Never silently drop malformed rows.
Always emit validation reports for malformed, missing, or filtered examples.
Any preprocessing script must:
- read from an input path
- write to an output path
- print a concise summary
- support a small-sample smoke mode

## Training rules

Every training stage must have:
- a config file
- a script entry point
- a smoke-test configuration
- deterministic seed handling
- output artifact directory structure
- minimal logging of loss and basic run metadata

SFT stage must support:
- loading processed instruction data
- running a small local smoke training loop
- saving adapter weights or checkpoints
- a basic inference sanity check

DPO stage must support:
- loading preference pairs
- validating chosen/rejected formatting
- running a small local smoke training loop
- saving adapter weights or checkpoints
- a basic post-train inference sanity check

## Evaluation rules

Evaluation must be lightweight and inspectable.
At minimum include:
- exact formatting checks
- JSON/schema validity if structured outputs are used
- simple lexical or task metrics when appropriate
- side-by-side generation dumps
- badcase sampling

Do not claim business wins or benchmark superiority without evidence.
If metrics are weak, report them honestly.

## Testing rules

Use pytest for unit and integration-style smoke tests where possible.

Minimum expectations by stage:
- schema and preprocessing tests
- config loading tests
- training smoke tests
- evaluation smoke tests

If heavy model execution is too expensive for CI, create mock-based or tiny-sample tests that still verify the code path.

## Required commands

When relevant, Codex should try these commands and adapt if the repo uses a different setup:

Environment:
- `python -V`
- `pip install -r requirements.txt`

Tests:
- `pytest -q`

Lint if configured:
- `ruff check .`
- `black --check .`

Data preprocessing example:
- `python scripts/prepare_sft_data.py --input data/raw/sft_sample.jsonl --output data/processed/sft_train.jsonl --smoke-test`

SFT smoke example:
- `python scripts/train_sft.py --config configs/sft_smoke.yaml`

DPO smoke example:
- `python scripts/train_dpo.py --config configs/dpo_smoke.yaml`

Eval smoke example:
- `python scripts/eval_model.py --config configs/eval_smoke.yaml`

Only run commands that exist in the repo. If a command is missing but required by the current stage, create it.

## Definition of done

A task is done only if:
1. the requested scope is implemented
2. affected commands or tests pass
3. docs or inline comments are updated where necessary
4. outputs are saved in predictable locations
5. the final summary states:
   - what changed
   - which tests ran
   - which metrics or artifacts were produced
   - known limitations

## Final response format for Codex

At the end of each task, provide:
1. Summary of changes
2. Files added or modified
3. Commands run
4. Test results
5. Artifacts produced
6. Remaining risks or next recommended step

## Safety and repo hygiene

Never commit secrets.
Never fabricate experimental results.
Never delete user data or unrelated files.
Never replace real training code with placeholder pseudocode unless the user explicitly asked for a scaffold only.
If a stub is unavoidable, mark it clearly with TODO and explain why.