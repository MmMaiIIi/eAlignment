#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-dpo}"

python scripts/prepare_pref.py --profile "${PROFILE}"
bash scripts/launch_dpo.sh "${PROFILE}"

