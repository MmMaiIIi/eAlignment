#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/llamafactory/dpo/smoke.yaml}"
CLI_BIN="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

CMD=("${CLI_BIN}" train "${CONFIG_PATH}")
echo "Launching DPO command: ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
