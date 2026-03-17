#!/usr/bin/env bash
set -euo pipefail

PROFILE_OR_CONFIG="${1:-smoke}"
CLI_BIN="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DRY_RUN="${DRY_RUN:-0}"

case "${PROFILE_OR_CONFIG}" in
  smoke)
    CONFIG_PATH="configs/sft_smoke.yaml"
    ;;
  sft)
    CONFIG_PATH="configs/sft.yaml"
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

USE_TORCHRUN="${FORCE_TORCHRUN:-0}"

if grep -Eq '^[[:space:]]*deepspeed[[:space:]]*:' "${CONFIG_PATH}"; then
  USE_TORCHRUN=1
fi

CMD=("${CLI_BIN}" train "${CONFIG_PATH}")

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  echo "Launching SFT (${PROFILE_OR_CONFIG}) command: FORCE_TORCHRUN=1 ${CMD[*]}"
else
  echo "Launching SFT (${PROFILE_OR_CONFIG}) command: ${CMD[*]}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  FORCE_TORCHRUN=1 "${CMD[@]}"
else
  "${CMD[@]}"
fi