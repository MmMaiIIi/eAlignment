#!/usr/bin/env bash
set -euo pipefail

PROFILE_OR_CONFIG="${1:-smoke}"
CLI_BIN="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DRY_RUN="${DRY_RUN:-0}"
SFT_SKIP_BEFORE_AFTER="${SFT_SKIP_BEFORE_AFTER:-0}"
SFT_COMPARE_MAX_NEW_TOKENS="${SFT_COMPARE_MAX_NEW_TOKENS:-96}"
SFT_COMPARE_TEMPERATURE="${SFT_COMPARE_TEMPERATURE:-0.0}"
SFT_COMPARE_TOP_P="${SFT_COMPARE_TOP_P:-1.0}"
SFT_COMPARE_REPETITION_PENALTY="${SFT_COMPARE_REPETITION_PENALTY:-1.0}"
SFT_EVAL_PROMPTS_PATH="${SFT_EVAL_PROMPTS_PATH:-data/synthetic/sft_eval_prompts.jsonl}"
RUN_TIMESTAMP="${SFT_RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${PROFILE_OR_CONFIG}" in
  smoke)
    CONFIG_PATH="configs/sft_smoke.yaml"
    ;;
  sft_lora_nodeepspeed_smoke)
    CONFIG_PATH="configs/sft_lora_nodeepspeed_smoke.yaml"
    ;;
  sft_plain_smoke)
    CONFIG_PATH="configs/sft_plain_smoke.yaml"
    ;;
  sft)
    CONFIG_PATH="configs/sft.yaml"
    ;;
  sft_lora_nodeepspeed)
    CONFIG_PATH="configs/sft_lora_nodeepspeed.yaml"
    ;;
  sft_plain)
    CONFIG_PATH="configs/sft_plain.yaml"
    ;;
  sft_qlora)
    CONFIG_PATH="configs/sft_qlora.yaml"
    ;;
  *)
    CONFIG_PATH="${PROFILE_OR_CONFIG}"
    ;;
esac

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

read_yaml_scalar() {
  local key="$1"
  local path="$2"
  grep -E "^[[:space:]]*${key}[[:space:]]*:" "${path}" | head -n 1 | sed -E "s/^[^:]+:[[:space:]]*//" || true
}

slugify() {
  local text="$1"
  text="$(echo "${text}" | tr '[:upper:]' '[:lower:]')"
  text="$(echo "${text}" | sed -E 's#[^a-z0-9]+#-#g; s#^-+##; s#-+$##')"
  if [[ -z "${text}" ]]; then
    text="unknown"
  fi
  echo "${text}"
}

USE_TORCHRUN="${FORCE_TORCHRUN:-0}"

if grep -Eq '^[[:space:]]*deepspeed[[:space:]]*:' "${CONFIG_PATH}"; then
  USE_TORCHRUN=1
fi

CMD=("${CLI_BIN}" train "${CONFIG_PATH}")
MODEL_NAME_OR_PATH="$(read_yaml_scalar model_name_or_path "${CONFIG_PATH}")"
MODEL_TAG="$(slugify "${MODEL_NAME_OR_PATH##*/}")"
PROFILE_TAG="$(slugify "${PROFILE_OR_CONFIG}")"
RUN_NAME="${SFT_RUN_NAME:-${RUN_TIMESTAMP}_${PROFILE_TAG}_${MODEL_TAG}}"
ARTIFACT_DIR="runs/artifacts/sft/${RUN_NAME}"

OUTPUT_DIR="$(read_yaml_scalar output_dir "${CONFIG_PATH}")"
DATASET_DIR="$(read_yaml_scalar dataset_dir "${CONFIG_PATH}")"
DATASET_NAME="$(read_yaml_scalar dataset "${CONFIG_PATH}")"
NUM_TRAIN_EPOCHS="$(read_yaml_scalar num_train_epochs "${CONFIG_PATH}")"
LEARNING_RATE="$(read_yaml_scalar learning_rate "${CONFIG_PATH}")"
SEED_VALUE="$(read_yaml_scalar seed "${CONFIG_PATH}")"
FINETUNING_TYPE="$(read_yaml_scalar finetuning_type "${CONFIG_PATH}")"
LORA_RANK="$(read_yaml_scalar lora_rank "${CONFIG_PATH}")"
LORA_ALPHA="$(read_yaml_scalar lora_alpha "${CONFIG_PATH}")"
LORA_DROPOUT="$(read_yaml_scalar lora_dropout "${CONFIG_PATH}")"
LORA_TARGET="$(read_yaml_scalar lora_target "${CONFIG_PATH}")"
LOGGING_STEPS="$(read_yaml_scalar logging_steps "${CONFIG_PATH}")"
SAVE_STEPS="$(read_yaml_scalar save_steps "${CONFIG_PATH}")"
REPORT_TO="$(read_yaml_scalar report_to "${CONFIG_PATH}")"
LOGGING_DIR="$(read_yaml_scalar logging_dir "${CONFIG_PATH}")"

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="outputs/sft/${PROFILE_TAG}"
fi
if [[ -z "${SEED_VALUE}" ]]; then
  SEED_VALUE="42"
fi

mkdir -p "${ARTIFACT_DIR}" "reports"
cp "${CONFIG_PATH}" "${ARTIFACT_DIR}/config_snapshot.yaml"

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  LAUNCH_COMMAND="FORCE_TORCHRUN=1 ${CMD[*]}"
else
  LAUNCH_COMMAND="${CMD[*]}"
fi
printf '%s\n' "${LAUNCH_COMMAND}" > "${ARTIFACT_DIR}/launch_command.txt"
echo "Launching SFT (${PROFILE_OR_CONFIG}) command: ${LAUNCH_COMMAND}"
echo "SFT run name: ${RUN_NAME}"
echo "SFT artifact dir: ${ARTIFACT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, training and artifact harvesting skipped."
  exit 0
fi

BEFORE_CACHE_PATH="${ARTIFACT_DIR}/before_outputs.jsonl"
BEFORE_AFTER_PATH="${ARTIFACT_DIR}/before_after.jsonl"
BEFORE_AFTER_MD="${ARTIFACT_DIR}/before_after.md"
CONSOLE_LOG_PATH="${ARTIFACT_DIR}/console.log"
TRAINING_LOG_PATH="${ARTIFACT_DIR}/training_log.jsonl"
TRAINER_STATE_SOURCE="${OUTPUT_DIR}/trainer_state.json"
TRAINER_STATE_COPY="${ARTIFACT_DIR}/trainer_state.json"
CHECKPOINT_POINTER_PATH="${ARTIFACT_DIR}/checkpoint_dirs.txt"
TENSORBOARD_POINTER_PATH="${ARTIFACT_DIR}/tensorboard_event_files.txt"
MANIFEST_PATH="${ARTIFACT_DIR}/run_manifest.json"
LOSS_CURVE_PNG="reports/loss_curve_${RUN_NAME}.png"
LOSS_CURVE_JSON="reports/loss_curve_${RUN_NAME}.json"
SUMMARY_PATH="reports/sft_run_summary_${RUN_NAME}.md"

if [[ "${SFT_SKIP_BEFORE_AFTER}" != "1" ]]; then
  if ! python scripts/compare_sft_generations.py \
    --mode before \
    --prompts "${SFT_EVAL_PROMPTS_PATH}" \
    --model "${MODEL_NAME_OR_PATH}" \
    --output "${BEFORE_CACHE_PATH}" \
    --max-new-tokens "${SFT_COMPARE_MAX_NEW_TOKENS}" \
    --temperature "${SFT_COMPARE_TEMPERATURE}" \
    --top-p "${SFT_COMPARE_TOP_P}" \
    --repetition-penalty "${SFT_COMPARE_REPETITION_PENALTY}" \
    --seed "${SEED_VALUE}"; then
    echo "Warning: base-model generation failed before training." >&2
  fi
fi

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  FORCE_TORCHRUN=1 "${CMD[@]}" 2>&1 | tee "${CONSOLE_LOG_PATH}"
else
  "${CMD[@]}" 2>&1 | tee "${CONSOLE_LOG_PATH}"
fi

if [[ -f "${TRAINER_STATE_SOURCE}" ]]; then
  cp "${TRAINER_STATE_SOURCE}" "${TRAINER_STATE_COPY}"
fi

TRAINING_LOG_SOURCE=""
if [[ -f "${OUTPUT_DIR}/trainer_log.jsonl" ]]; then
  cp "${OUTPUT_DIR}/trainer_log.jsonl" "${TRAINING_LOG_PATH}"
  TRAINING_LOG_SOURCE="${OUTPUT_DIR}/trainer_log.jsonl"
elif [[ -f "${TRAINER_STATE_SOURCE}" ]]; then
  python - "${TRAINER_STATE_SOURCE}" "${TRAINING_LOG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
state = json.loads(state_path.read_text(encoding="utf-8"))
history = state.get("log_history", [])
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8", newline="\n") as fh:
    for row in history:
        if isinstance(row, dict):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
  TRAINING_LOG_SOURCE="${TRAINER_STATE_SOURCE}#log_history"
else
  python - "${TRAINING_LOG_PATH}" "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
output_dir = sys.argv[2]
output_path.parent.mkdir(parents=True, exist_ok=True)
row = {
    "warning": "No trainer_log.jsonl or trainer_state.json found after training.",
    "output_dir": output_dir,
}
output_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
PY
fi

: > "${CHECKPOINT_POINTER_PATH}"
: > "${TENSORBOARD_POINTER_PATH}"
if [[ -d "${OUTPUT_DIR}" ]]; then
  while IFS= read -r checkpoint_path; do
    if [[ -n "${checkpoint_path}" ]]; then
      printf '%s\n' "${checkpoint_path}" >> "${CHECKPOINT_POINTER_PATH}"
    fi
  done < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort)

  while IFS= read -r event_path; do
    if [[ -n "${event_path}" ]]; then
      printf '%s\n' "${event_path}" >> "${TENSORBOARD_POINTER_PATH}"
    fi
  done < <(find "${OUTPUT_DIR}" -type f -name 'events.out.tfevents*' | sort)
fi

python scripts/plot_loss.py \
  --run-name "${RUN_NAME}" \
  --training-log "${TRAINING_LOG_PATH}" \
  --trainer-state "${TRAINER_STATE_COPY}" \
  --output-png "${LOSS_CURVE_PNG}" \
  --output-json "${LOSS_CURVE_JSON}"

BEFORE_AFTER_ARTIFACT="${BEFORE_AFTER_MD}"
if [[ "${SFT_SKIP_BEFORE_AFTER}" == "1" ]]; then
  cat > "${BEFORE_AFTER_MD}" <<EOF
# Before/After Comparison Skipped

- reason: SFT_SKIP_BEFORE_AFTER=1
- prompts_path: ${SFT_EVAL_PROMPTS_PATH}
- before_model: ${MODEL_NAME_OR_PATH}
- after_model: ${OUTPUT_DIR}
EOF
elif [[ -f "${BEFORE_CACHE_PATH}" ]]; then
  if python scripts/compare_sft_generations.py \
    --mode after \
    --prompts "${SFT_EVAL_PROMPTS_PATH}" \
    --model "${OUTPUT_DIR}" \
    --before-cache "${BEFORE_CACHE_PATH}" \
    --output "${BEFORE_AFTER_PATH}" \
    --max-new-tokens "${SFT_COMPARE_MAX_NEW_TOKENS}" \
    --temperature "${SFT_COMPARE_TEMPERATURE}" \
    --top-p "${SFT_COMPARE_TOP_P}" \
    --repetition-penalty "${SFT_COMPARE_REPETITION_PENALTY}" \
    --seed "${SEED_VALUE}"; then
    BEFORE_AFTER_ARTIFACT="${BEFORE_AFTER_PATH}"
    rm -f "${BEFORE_AFTER_MD}"
  else
    cat > "${BEFORE_AFTER_MD}" <<EOF
# Before/After Comparison Failed

- prompts_path: ${SFT_EVAL_PROMPTS_PATH}
- before_cache_path: ${BEFORE_CACHE_PATH}
- before_model: ${MODEL_NAME_OR_PATH}
- after_model: ${OUTPUT_DIR}
- note: Generation comparison command failed. See ${CONSOLE_LOG_PATH} for training logs.
EOF
  fi
else
  cat > "${BEFORE_AFTER_MD}" <<EOF
# Before/After Comparison Missing

- prompts_path: ${SFT_EVAL_PROMPTS_PATH}
- before_model: ${MODEL_NAME_OR_PATH}
- after_model: ${OUTPUT_DIR}
- note: Missing before-cache artifact, so after comparison was not generated.
EOF
fi

cat > "${SUMMARY_PATH}" <<EOF
# SFT Run Summary: ${RUN_NAME}

- profile_or_config: ${PROFILE_OR_CONFIG}
- source_config: ${CONFIG_PATH}
- output_dir: ${OUTPUT_DIR}
- model_name_or_path: ${MODEL_NAME_OR_PATH}
- dataset: ${DATASET_NAME}
- dataset_dir: ${DATASET_DIR}
- num_train_epochs: ${NUM_TRAIN_EPOCHS}
- learning_rate: ${LEARNING_RATE}
- finetuning_type: ${FINETUNING_TYPE}
- lora_rank: ${LORA_RANK}
- lora_alpha: ${LORA_ALPHA}
- lora_dropout: ${LORA_DROPOUT}
- lora_target: ${LORA_TARGET}
- logging_steps: ${LOGGING_STEPS}
- save_steps: ${SAVE_STEPS}
- report_to: ${REPORT_TO}
- logging_dir: ${LOGGING_DIR}
- launch_command_file: ${ARTIFACT_DIR}/launch_command.txt
- config_snapshot_file: ${ARTIFACT_DIR}/config_snapshot.yaml
- training_log_file: ${TRAINING_LOG_PATH}
- trainer_state_file: ${TRAINER_STATE_COPY}
- checkpoint_pointer_file: ${CHECKPOINT_POINTER_PATH}
- tensorboard_pointer_file: ${TENSORBOARD_POINTER_PATH}
- loss_curve_png: ${LOSS_CURVE_PNG}
- loss_curve_json: ${LOSS_CURVE_JSON}
- before_after_artifact: ${BEFORE_AFTER_ARTIFACT}
- run_manifest: ${MANIFEST_PATH}
EOF

export RUN_NAME PROFILE_OR_CONFIG CONFIG_PATH OUTPUT_DIR MODEL_NAME_OR_PATH DATASET_DIR DATASET_NAME
export NUM_TRAIN_EPOCHS LEARNING_RATE SEED_VALUE FINETUNING_TYPE LORA_RANK LORA_ALPHA LORA_DROPOUT LORA_TARGET
export LOGGING_STEPS SAVE_STEPS REPORT_TO LOGGING_DIR ARTIFACT_DIR MANIFEST_PATH
export CONSOLE_LOG_PATH TRAINING_LOG_PATH TRAINER_STATE_COPY CHECKPOINT_POINTER_PATH TENSORBOARD_POINTER_PATH
export LOSS_CURVE_PNG LOSS_CURVE_JSON SUMMARY_PATH SFT_EVAL_PROMPTS_PATH BEFORE_CACHE_PATH BEFORE_AFTER_ARTIFACT
export TRAINING_LOG_SOURCE LAUNCH_COMMAND RUN_TIMESTAMP
python - <<'PY'
import json
import os
from pathlib import Path


def read_nonempty_lines(path_str: str) -> list[str]:
    path = Path(path_str)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def existing_or_none(path_str: str) -> str | None:
    path = Path(path_str)
    return path.as_posix() if path.exists() else None


manifest = {
    "run_name": os.environ["RUN_NAME"],
    "created_at_utc": os.environ["RUN_TIMESTAMP"],
    "profile_or_config": os.environ["PROFILE_OR_CONFIG"],
    "source_config_path": Path(os.environ["CONFIG_PATH"]).as_posix(),
    "artifacts_dir": Path(os.environ["ARTIFACT_DIR"]).as_posix(),
    "launch_command": os.environ["LAUNCH_COMMAND"],
    "paths": {
        "launch_command_file": Path(os.environ["ARTIFACT_DIR"], "launch_command.txt").as_posix(),
        "config_snapshot_file": Path(os.environ["ARTIFACT_DIR"], "config_snapshot.yaml").as_posix(),
        "console_log_file": Path(os.environ["CONSOLE_LOG_PATH"]).as_posix(),
        "training_log_file": Path(os.environ["TRAINING_LOG_PATH"]).as_posix(),
        "trainer_state_file": existing_or_none(os.environ["TRAINER_STATE_COPY"]),
        "checkpoint_pointer_file": Path(os.environ["CHECKPOINT_POINTER_PATH"]).as_posix(),
        "tensorboard_pointer_file": Path(os.environ["TENSORBOARD_POINTER_PATH"]).as_posix(),
        "loss_curve_png": Path(os.environ["LOSS_CURVE_PNG"]).as_posix(),
        "loss_curve_json": existing_or_none(os.environ["LOSS_CURVE_JSON"]),
        "summary_markdown": Path(os.environ["SUMMARY_PATH"]).as_posix(),
        "before_cache": existing_or_none(os.environ["BEFORE_CACHE_PATH"]),
        "before_after": existing_or_none(os.environ["BEFORE_AFTER_ARTIFACT"]),
    },
    "runtime": {
        "model_name_or_path": os.environ["MODEL_NAME_OR_PATH"],
        "dataset": os.environ["DATASET_NAME"],
        "dataset_dir": os.environ["DATASET_DIR"],
        "output_dir": os.environ["OUTPUT_DIR"],
        "num_train_epochs": os.environ["NUM_TRAIN_EPOCHS"],
        "learning_rate": os.environ["LEARNING_RATE"],
        "seed": os.environ["SEED_VALUE"],
        "finetuning_type": os.environ["FINETUNING_TYPE"],
        "lora_rank": os.environ["LORA_RANK"],
        "lora_alpha": os.environ["LORA_ALPHA"],
        "lora_dropout": os.environ["LORA_DROPOUT"],
        "lora_target": os.environ["LORA_TARGET"],
        "logging_steps": os.environ["LOGGING_STEPS"],
        "save_steps": os.environ["SAVE_STEPS"],
        "report_to": os.environ["REPORT_TO"],
        "logging_dir": os.environ["LOGGING_DIR"],
        "training_log_source": os.environ["TRAINING_LOG_SOURCE"],
    },
    "checkpoints": {
        "directories": read_nonempty_lines(os.environ["CHECKPOINT_POINTER_PATH"]),
    },
    "tensorboard": {
        "event_files": read_nonempty_lines(os.environ["TENSORBOARD_POINTER_PATH"]),
    },
    "comparison": {
        "prompts_path": Path(os.environ["SFT_EVAL_PROMPTS_PATH"]).as_posix(),
        "before_cache_path": existing_or_none(os.environ["BEFORE_CACHE_PATH"]),
        "before_after_artifact": existing_or_none(os.environ["BEFORE_AFTER_ARTIFACT"]),
    },
}

manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "SFT artifact bundle ready: ${ARTIFACT_DIR}"
