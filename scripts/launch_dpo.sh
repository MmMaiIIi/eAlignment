#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/llamafactory/dpo/qwen3_8b_lora_dpo.yaml}"

echo "Launching DPO with config: ${CONFIG_PATH}"
llamafactory-cli train "${CONFIG_PATH}"
