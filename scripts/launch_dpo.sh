#!/usr/bin/env bash
set -euo pipefail

PROFILE_OR_CONFIG="${1:-smoke}"
CLI_BIN="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DRY_RUN="${DRY_RUN:-0}"

case "${PROFILE_OR_CONFIG}" in
  smoke)
    CONFIG_PATH="configs/dpo_smoke.yaml"
    ;;
  dpo)
    CONFIG_PATH="configs/dpo.yaml"
    ;;
  *)
    CONFIG_PATH="${PROFILE_OR_CONFIG}"
    ;;
esac

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

CMD=("${CLI_BIN}" train "${CONFIG_PATH}")
echo "Launching DPO (${PROFILE_OR_CONFIG}) command: ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
