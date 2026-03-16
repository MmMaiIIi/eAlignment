# eAlignment – Project Status (Stage 0)

## Overview

This repository currently contains the **Stage-0 scaffold** for the alignment training project.
The goal of this stage is to establish the **engineering structure, configuration system, sample datasets, and testing framework** required for future implementation of the alignment pipeline.

At this stage, **no training or evaluation logic has been implemented yet**. The repository only provides the foundational infrastructure needed to build the alignment system in later stages.

---

## Project Structure

The repository follows a typical machine learning project layout designed for alignment training experiments.

```text
eAlignment/
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ samples/
├─ src/
│  ├─ config/
│  ├─ data/
│  ├─ training/
│  ├─ eval/
│  └─ utils/
├─ scripts/
├─ tests/
├─ artifacts/
└─ reports/
```

### Key Directories

**configs/**
Configuration files used for training and evaluation experiments.

**data/**
Dataset storage.

* `raw/` – original datasets
* `processed/` – preprocessed datasets
* `samples/` – synthetic smoke-test datasets

**src/**
Source code for the alignment system.

* `data/` – dataset loading utilities
* `training/` – future SFT and DPO training code
* `eval/` – evaluation logic
* `utils/` – common utilities such as configuration loading

**scripts/**
Entry points for data processing, training, and evaluation scripts (to be implemented in later stages).

**artifacts/**
Model checkpoints and experiment outputs.

**reports/**
Evaluation reports and experiment summaries.

---

## Configuration Files

Three **smoke-test configurations** are included to verify that configuration loading works correctly.

```text
configs/
├─ sft_smoke.yaml
├─ dpo_smoke.yaml
└─ eval_smoke.yaml
```

These files simulate configurations for:

* **SFT (Supervised Fine-Tuning)**
* **DPO (Direct Preference Optimization)**
* **Evaluation**

At the moment, these configurations are only used for **testing the configuration loading pipeline**.

---

## Sample Datasets

Synthetic sample datasets are included for testing the data loading pipeline.

### SFT Sample

File:

```text
data/samples/sft_sample.jsonl
```

Structure:

```json
{
  "instruction": "...",
  "response": "..."
}
```

### DPO Sample

File:

```text
data/samples/dpo_sample.jsonl
```

Structure:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

These datasets are **not intended for real training**. They are only used for smoke tests.

---

## Utility Modules

Two utility modules are implemented to support the future training pipeline.

### Config Loader

```text
src/utils/config_loader.py
```

Responsible for:

* Loading YAML configuration files
* Converting configurations into Python dictionaries

### Data IO

```text
src/data/io.py
```

Responsible for:

* Reading JSONL datasets
* Providing simple dataset loading helpers

These modules serve as the **foundation for the future training and evaluation pipelines**.

---

## Testing

Basic **pytest smoke tests** are included to verify that the repository structure works correctly.

Tests include:

```text
tests/
├─ test_imports.py
├─ test_configs.py
└─ test_samples.py
```

The tests verify:

1. All core modules can be imported successfully
2. Configuration files can be loaded correctly
3. Sample datasets can be read without errors

### Test Result

```text
pytest -q
6 passed in 0.10s
```

This confirms that the repository scaffold is functioning as expected.

---

## Environment

Python version used during setup:

```text
Python 3.11.9
```

Dependencies are managed through:

```text
requirements.txt
```

---

## Artifacts

No training artifacts have been generated yet.

Current artifacts consist only of:

* configuration files
* sample datasets
* utility modules
* passing smoke tests

---

## Deferred to Stage 1

The following components will be implemented in the next development stage:

* Data schema design and validation pipeline
* Full preprocessing scripts for raw → processed datasets
* Runnable training scripts for:

  * **SFT**
  * **DPO**
* Training loops and smoke training runs
* Model checkpoint generation
* Evaluation scripts
* Experiment report generation

---

## Current Status

The project is currently a **clean engineering scaffold for an alignment training system**.

Completed:

* Repository structure
* Configuration system
* Sample datasets
* Utility modules
* Smoke tests

Not yet implemented:

* Data preprocessing
* SFT training
* DPO training
* Model checkpoints
* Evaluation pipeline

The repository is now ready for **Stage-1 implementation of the alignment training pipeline**.
