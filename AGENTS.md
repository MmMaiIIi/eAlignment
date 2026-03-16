# AGENTS.md

## Project Identity

This repository implements Project 1: an interview-ready LLM alignment project for Chinese e-commerce customer support.

The repository is intentionally framework-first, not framework-reimplementation-first.

We will use existing, mature open-source tooling as much as possible:

- Qwen3-8B as the base model
- LLaMA-Factory as the primary training framework
- Transformers / TRL / PEFT as the underlying fine-tuning stack
- DeepSpeed ZeRO-2 as the training infrastructure option

This repository should NOT build a custom SFT/DPO training framework from scratch.
Instead, it should provide a clean adaptation layer around existing frameworks, focused on:

- dataset preparation
- domain-specific formatting
- experiment configs
- launch scripts
- evaluation
- badcase analysis
- reports and documentation

---

## Core Goal

Build a complete and reproducible alignment project for Chinese e-commerce customer support that covers:

- SFT
- DPO
- evaluation
- ablation
- badcase analysis

while minimizing unnecessary custom training code.

The final project should look like something an applied LLM intern or junior LLM engineer would actually build:
practical, reproducible, and easy to explain.

---

## Main Principle

Use existing frameworks directly.
Do not reinvent wheels.

Specifically:

- Do not implement a custom trainer if LLaMA-Factory already supports the workflow.
- Do not implement custom LoRA/QLoRA injection logic.
- Do not implement custom DPO loss or trainer logic.
- Do not implement custom DeepSpeed orchestration.
- Do not create a giant internal framework.

Custom code is allowed only when it adds project-specific value, such as:

- data cleaning and transformation
- schema validation
- domain-specific evaluation
- experiment summaries
- reporting

---

## Framework Strategy

### Primary framework
Use LLaMA-Factory as the main training entrypoint for:

- SFT
- DPO
- LoRA / QLoRA
- DeepSpeed integration
- model export where needed

### Supporting libraries
Use these as standard dependencies and references:

- transformers
- trl
- peft
- accelerate
- deepspeed
- datasets
- evaluate
- pandas
- pyyaml

### Boundary
This repository should be a project repo around LLaMA-Factory, not a replacement for LLaMA-Factory.

---

## Target Domain

Chinese e-commerce customer support, including at minimum:

- returns_refunds
- shipping_logistics
- product_specs
- order_modification
- after_sales
- complaint_soothing

Typical tasks include:

- handling refund/return requests
- explaining delivery status and delays
- answering product detail questions
- modifying or cancelling orders
- responding to emotionally frustrated customers
- producing policy-compliant support replies

---

## Repository Design

Prefer a structure like this:

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── llamafactory/
│   │   ├── sft/
│   │   ├── dpo/
│   │   └── deepspeed/
│   ├── data/
│   └── eval/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── synthetic/
├── scripts/
│   ├── prepare_sft_data.py
│   ├── prepare_dpo_data.py
│   ├── launch_sft.sh
│   ├── launch_dpo.sh
│   ├── export_model.sh
│   ├── run_eval.py
│   └── summarize_badcases.py
├── src/
│   ├── data/
│   ├── eval/
│   ├── prompts/
│   └── utils/
├── reports/
│   ├── templates/
│   ├── experiments/
│   └── badcases/
└── tests/
    ├── test_configs.py
    ├── test_data_pipeline.py
    ├── test_eval_pipeline.py
    └── test_smoke_assets.py
````

Important:
Do not create a large `src/train/` framework unless there is a very small helper that is truly necessary.
Training should mostly be driven by LLaMA-Factory configs and wrapper scripts.

---

## What This Repo Owns

This repository should own:

1. Data schemas
2. Data conversion and validation scripts
3. LLaMA-Factory-ready SFT and DPO datasets
4. LLaMA-Factory config files
5. Wrapper scripts for launching experiments
6. Evaluation code
7. Ablation definitions
8. Badcase analysis workflow
9. Reports and interview-facing documentation

---

## What This Repo Should Not Own

This repository should not own:

* a custom trainer stack
* a custom LoRA framework
* a custom DPO implementation
* a custom distributed training engine
* copied source code from LLaMA-Factory, TRL, or PEFT

If integration with an upstream framework is needed, prefer:

* documented installation
* wrapper scripts
* config files
* light adapters

Do not vendor large framework code into this repo.

---

## Assumption About LLaMA-Factory

Assume the user will install LLaMA-Factory separately or make `llamafactory-cli` available in the environment.

This repo should:

* document that dependency clearly
* generate configs and data that are directly usable by LLaMA-Factory
* provide launch scripts around it

Do not try to copy LLaMA-Factory internals into this project.

---

## Dataset Design

### SFT dataset

Store normalized instruction/chat examples for customer support.

Preferred normalized JSONL example:

```json
{
  "id": "sft_0001",
  "category": "returns_refunds",
  "system": "You are a professional e-commerce customer support assistant.",
  "instruction": "The customer says the product arrived damaged and asks how to request a return.",
  "input": "",
  "output": "I am sorry this happened. Please provide your order number and a photo of the damaged item. We will help you arrange a return or replacement as soon as possible."
}
````

### Preference dataset

Store prompt, chosen, rejected.

Preferred normalized JSONL example:

```json
{
  "id": "pref_0001",
  "category": "complaint_soothing",
  "prompt": "Customer: Your delivery is late again. This is unacceptable.",
  "chosen": "I am sorry for the delay and understand your frustration. Please send your order number and I will check the latest delivery status and available support options for you.",
  "rejected": "Please wait patiently. Delays happen."
}
````

All custom scripts should convert raw/internal formats into the formats required by LLaMA-Factory or closely aligned with it.

---

## Evaluation Philosophy

This is an applied project.
Evaluation should emphasize business-facing quality, not only train loss.

At minimum, support some combination of:

* category-level evaluation
* response quality checks
* actionability checks
* politeness/tone checks
* policy compliance checks
* badcase collection

If proxy metrics are used, clearly label them as proxy metrics.

Do not invent benchmark gains.

---

## Stage Plan

### Stage 0

Scaffold the repo around LLaMA-Factory usage.
Set up dependencies, configs, synthetic data, smoke assets, and documentation.

### Stage 1

Implement data schemas, validation, normalization, and dataset conversion scripts.

### Stage 2

Add SFT experiment configs and wrapper launch scripts using LLaMA-Factory.

### Stage 3

Add evaluation pipeline, model comparison workflow, and badcase collection.

### Stage 4

Add preference-data construction, validation, and DPO configs using LLaMA-Factory.

### Stage 5

Add ablation workflow, final reports, and interview-ready project packaging.

Do not skip directly to full implementation.
Build the project stage by stage.

---

## Coding Standards

* Prefer small, explicit Python utilities.
* Use YAML for configs.
* Keep scripts easy to run from the command line.
* Avoid heavy abstractions.
* Avoid hidden state.
* Keep functions short and typed where practical.
* Document assumptions.

This repository should remain compact and legible.

---

## Testing Requirements

Each stage should leave behind useful tests.

Minimum expectations:

* config loading test
* synthetic data readability test
* schema validation test
* conversion pipeline smoke test
* eval pipeline smoke test

If a full train run is too expensive, rely on:

* synthetic examples
* config validation
* no-crash command generation
* output schema validation

Tests must not depend on private or unavailable data.

---

## Documentation Requirements

README should clearly explain:

* project goal
* why LLaMA-Factory is used
* repository structure
* setup
* data preparation
* SFT workflow
* DPO workflow
* evaluation workflow
* experiment outputs
* limitations
* interview talking points

Longer experiment summaries should go into `reports/`.

---

## Guardrails for Codex

Before editing:

1. Read AGENTS.md.
2. Inspect the current repository tree.
3. Preserve working files unless change is necessary.

When implementing:

1. Prefer framework integration over custom reimplementation.
2. Add only project-specific glue code.
3. Keep changes incremental.
4. Run the narrowest useful checks after each change.
5. Update documentation when behavior changes.
6. Never fabricate experiment results.
7. Clearly mark verified vs unverified steps.

Always end with:

* summary of changes
* files changed
* commands run
* current status
* remaining TODOs or risks

---

## Definition of Done

A stage is done only if:

* repo structure is coherent
* configs exist
* docs are updated
* smoke checks pass
* generated assets are valid
* there is no fake completeness

---

## Final Outcome

The final repository should demonstrate:

* clean domain data preparation
* practical use of LLaMA-Factory for SFT and DPO
* meaningful evaluation and comparison
* disciplined ablation thinking
* clear badcase analysis
* a polished, interview-ready engineering story

````