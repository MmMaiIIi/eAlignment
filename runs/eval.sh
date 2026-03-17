#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-eval}"

python scripts/eval.py --profile "${PROFILE}"
python scripts/badcases.py --profile "${PROFILE}" --top-k 10

