#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/llamafactory/sft/qwen3_8b_lora_sft.yaml}"

echo "Launching SFT with config: ${CONFIG_PATH}"
llamafactory-cli train "${CONFIG_PATH}"
