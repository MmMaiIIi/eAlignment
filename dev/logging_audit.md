# Experiment Logging Audit

Verdict: partially sufficient

## Scope

Audit date: 2026-03-20

Evidence was gathered from two separate buckets:

1. Implemented in code/config
2. Artifacts currently present on disk

The main execution path audited was `runs/ -> scripts/ -> align/`, with supporting checks in `configs/`, `data/`, `reports/`, and `README.md`.

## Summary

This repo is strong on preprocessing and offline evaluation traceability:

- dataset metadata is generated and present
- rejected preprocessing rows are logged and present
- DPO quality reports are generated and present
- before/after offline inference comparisons are generated and present
- eval summaries, per-sample comparisons, badcases, and markdown reports are generated and present

This repo is weak on training-runtime logging completeness inside the repo itself:

- training configs exist and set `logging_steps`, `save_steps`, `plot_loss`, and `output_dir`
- launch scripts call `llamafactory-cli train ...`
- but there is no repo-local code or script wiring for persisted plain logs
- there is no explicit repo-local wiring for TensorBoard event files
- there is no explicit repo-local handling of `trainer_state.json`
- there are currently no `outputs/` training artifact directories on disk
- there are currently no observed checkpoint directories, `trainer_state.json`, or `events.out.tfevents*` files on disk

Because the repo defines where training artifacts should go but does not currently retain or verify those training artifacts in this workspace, the end-to-end logging story is only partially sufficient.

## Evidence By Topic

### 1. Training configs

Implemented in code/config:

- [configs/sft.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft.yaml):23 sets `logging_steps: 10`
- [configs/sft.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft.yaml):24 sets `save_steps: 200`
- [configs/sft.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft.yaml):25 sets `plot_loss: true`
- [configs/sft.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft.yaml):26 sets `output_dir: outputs/sft/qwen3_8b_lora`
- [configs/sft_qlora.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft_qlora.yaml):25-28 contain the same logging/checkpoint/output pattern for QLoRA
- [configs/dpo.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/dpo.yaml):24-27 contain the same logging/checkpoint/output pattern for DPO
- [configs/sft_smoke.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft_smoke.yaml):17-20 and [configs/dpo_smoke.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/dpo_smoke.yaml):18-21 do the same for smoke runs

Artifacts currently present on disk:

- The config files above are present on disk.

Assessment:

- Present and concrete.

### 2. Launch scripts

Implemented in code/config:

- [runs/sft.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/runs/sft.sh):7-8 runs preprocessing then `bash scripts/launch_sft.sh`
- [runs/dpo.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/runs/dpo.sh):6-7 runs preference prep then `bash scripts/launch_dpo.sh`
- [runs/eval.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/runs/eval.sh):6-7 runs evaluation then badcase extraction
- [runs/smoke.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/runs/smoke.sh):4-11 covers data prep, dry-run training launches, eval, and badcases
- [scripts/launch_sft.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_sft.sh):34 builds `CMD=("${CLI_BIN}" train "${CONFIG_PATH}")`
- [scripts/launch_sft.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_sft.sh):37-39 echoes the command to stdout
- [scripts/launch_dpo.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_dpo.sh):25-26 builds and echoes `llamafactory-cli train ...`

Artifacts currently present on disk:

- All launch scripts are present on disk.

Assessment:

- Present and concrete, but they only launch training; they do not capture persistent log files.

### 3. `output_dir` usage

Implemented in code/config:

- SFT output roots are configured in [configs/sft.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft.yaml):26, [configs/sft_qlora.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft_qlora.yaml):28, and [configs/sft_smoke.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft_smoke.yaml):20
- DPO output roots are configured in [configs/dpo.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/dpo.yaml):27 and [configs/dpo_smoke.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/dpo_smoke.yaml):21
- Eval output root is configured in [configs/eval.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/eval.yaml):7 and mirrored in [configs/profiles.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/profiles.yaml):20, :40, :60, :80, :100
- [scripts/eval.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/eval.py):26-31 resolves and passes the eval `output_dir`
- [align/eval.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/eval.py):286 creates the eval output directory before writing artifacts

Artifacts currently present on disk:

- `reports/experiments/latest_eval/` exists and contains eval outputs
- `outputs/` does not exist in the workspace at audit time

Assessment:

- Eval `output_dir` is implemented and populated.
- Training `output_dir` is configured but no current training output tree is present.

### 4. Checkpoint saving

Implemented in code/config:

- `save_steps` is configured in [configs/sft.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft.yaml):24, [configs/sft_qlora.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft_qlora.yaml):26, [configs/dpo.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/dpo.yaml):25, [configs/sft_smoke.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/sft_smoke.yaml):18, and [configs/dpo_smoke.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/dpo_smoke.yaml):19
- Training is delegated to LLaMA-Factory via [scripts/launch_sft.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_sft.sh):34 and [scripts/launch_dpo.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_dpo.sh):25

Artifacts currently present on disk:

- No `checkpoint-*` directories were found anywhere in this repo workspace.
- No `outputs/` directory was found.

Assessment:

- Checkpoint saving is configured indirectly through upstream trainer settings.
- No current checkpoint artifacts are present in this workspace.

### 5. `trainer_state.json`

Implemented in code/config:

- No repo-local code or config mentions `trainer_state.json`.
- A repo-wide search for `trainer_state.json` returned no matches in `configs/`, `scripts/`, `align/`, `runs/`, or docs.

Artifacts currently present on disk:

- No `trainer_state.json` file was found anywhere in this repo workspace.

Assessment:

- Not implemented in repo-local code.
- Not present on disk.

### 6. TensorBoard event files

Implemented in code/config:

- Training configs include `logging_steps`, but there is no explicit `report_to`, `logging_dir`, or `tensorboard` setting in repo configs.
- A repo-wide scan of `configs/*.yaml`, `runs/*.sh`, and `scripts/*.sh` found no `tensorboard`, `logging_dir`, `report_to`, or `events.out.tfevents` wiring.

Artifacts currently present on disk:

- No `events.out.tfevents*` files were found anywhere in this repo workspace.

Assessment:

- No explicit TensorBoard integration is implemented in this repo.
- No TensorBoard event artifacts are currently present.

### 7. Plain logs

Implemented in code/config:

- [scripts/launch_sft.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_sft.sh):37-39 and [scripts/launch_dpo.sh](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/launch_dpo.sh):26 echo launch commands to stdout
- A scan of `runs/*.sh` and `scripts/*.sh` found no `tee`, `.log`, `2>&1`, or other file-based log capture

Artifacts currently present on disk:

- No repo-owned training/eval plain log files were found for the current workflow
- The only `.txt` files found were unrelated legacy/vendor data under `data/JDDC-Baseline-Seq2Seq-master/` plus `requirements.txt`

Assessment:

- Human-readable stdout exists during execution, but there is no persistent plain-log capture for the repo's current run path.

### 8. Dataset metadata

Implemented in code/config:

- [scripts/prepare_data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/prepare_data.py):22 and :33 wire `dataset_info.json`
- [scripts/prepare_pref.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/prepare_pref.py):23 and :34 wire `dataset_info.json`
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):513 defines `_write_dataset_info`
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):602 and :656 write dataset metadata during SFT and DPO preparation
- [README.md](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/README.md):143 and :158 document `data/processed/dataset_info.json`

Artifacts currently present on disk:

- `data/processed/dataset_info.json` exists
- `data/processed/jddc_relaxed_check/dataset_info.json` also exists

Assessment:

- Fully present.

### 9. Preprocessing rejected logs

Implemented in code/config:

- [scripts/prepare_data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/prepare_data.py):21 and :32 wire `sft_rejected.jsonl`
- [scripts/prepare_pref.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/prepare_pref.py):21 and :32 wire `dpo_rejected.jsonl`
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):225 defines the rejected-row payload
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):600 and :652 write rejected rows to disk
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):611 returns an SFT rejection summary

Artifacts currently present on disk:

- `data/interim/sft_rejected.jsonl` exists
- `data/interim/dpo_rejected.jsonl` exists
- `data/interim/jddc_relaxed_rejected.jsonl` exists
- Example live SFT rejection rows contain line number, source, error list, and raw payload
- Example live DPO rejection rows contain invalid identical chosen/rejected responses and empty chosen text

Assessment:

- Fully present.

### 10. Quality reports

Implemented in code/config:

- [scripts/prepare_pref.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/prepare_pref.py):22 and :33 wire the quality report path
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):461 defines `_quality_report`
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):654-655 compute and write the report
- [align/data.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/data.py):667 returns the report path in the summary
- [README.md](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/README.md):146 documents `data/interim/dpo_quality_report.json`

Artifacts currently present on disk:

- `data/interim/dpo_quality_report.json` exists
- Current report includes totals, valid/rejected counts, issue counts, category distribution, imbalance checks, and duplicate-pattern checks

Assessment:

- Fully present.

### 11. Saved before/after inference comparisons

Implemented in code/config:

- [configs/eval.yaml](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/configs/eval.yaml):1-2 points to base and tuned prediction files
- [scripts/eval.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/scripts/eval.py):14 describes the command as offline evaluation and comparison
- [align/eval.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/eval.py):83-97 aligns `base_response` and `tuned_response`
- [align/eval.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/eval.py):154 stores `comparison_label`
- [align/eval.py](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/align/eval.py):287-290 write `summary.json`, `per_sample.jsonl`, `badcases.jsonl`, and `report.md`
- [README.md](/d:/Workspace/LLM%20Intern/PJ1/eAlignment/README.md):201-204 documents those eval outputs

Artifacts currently present on disk:

- `reports/experiments/latest_eval/summary.json` exists
- `reports/experiments/latest_eval/per_sample.jsonl` exists
- `reports/experiments/latest_eval/badcases.jsonl` exists
- `reports/experiments/latest_eval/report.md` exists
- `reports/badcases/latest_badcases.md` exists
- Current `per_sample.jsonl` rows contain both `base.response` and `tuned.response`, plus `delta_proxy_score` and `comparison_label`

Assessment:

- Fully present for offline comparison artifacts.

## Implemented Vs Present On Disk

### Implemented in code/config

Implemented:

- training configs
- launch scripts
- eval `output_dir`
- training `output_dir` declarations
- checkpoint cadence configuration via `save_steps`
- dataset metadata generation
- preprocessing rejected logs
- DPO quality reports
- offline before/after inference comparison outputs
- badcase markdown generation

Not explicitly implemented:

- persistent plain-log capture for training/eval commands
- explicit TensorBoard wiring
- explicit `trainer_state.json` handling or verification

### Artifacts currently present on disk

Present:

- processed datasets
- dataset metadata
- rejected preprocessing logs
- DPO quality report
- eval summary/per-sample/badcase/report artifacts
- badcase markdown summary

Absent:

- `outputs/` training artifact tree
- `checkpoint-*` directories
- `trainer_state.json`
- `events.out.tfevents*`
- persistent plain logs for the current run path

## Final Verdict

Verdict: partially sufficient

Reason:

The repo has solid provenance for preprocessing and offline evaluation, including rejected-row logs, metadata, quality checks, and saved comparison artifacts. But for true end-to-end experiment logging completeness, the training stage is under-specified at the repo level and unverified on disk: there are configured output roots and save cadence settings, yet no retained checkpoints, no `trainer_state.json`, no TensorBoard event files, and no persistent plain logs in the current workspace.
