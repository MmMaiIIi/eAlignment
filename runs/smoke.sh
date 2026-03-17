#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --profile smoke
python scripts/prepare_pref.py --profile smoke

DRY_RUN=1 bash scripts/launch_sft.sh smoke
DRY_RUN=1 bash scripts/launch_dpo.sh smoke

python scripts/eval.py --profile smoke
python scripts/badcases.py --profile smoke --top-k 5

echo "Smoke workflow completed."

