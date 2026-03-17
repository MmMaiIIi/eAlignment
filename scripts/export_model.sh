#!/usr/bin/env bash
set -euo pipefail

PROFILE_OR_CONFIG="${1:-sft}"
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

CMD=("${CLI_BIN}" export "${CONFIG_PATH}")
echo "Export command: ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
