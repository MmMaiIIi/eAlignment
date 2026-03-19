
#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-sft}"

python scripts/prepare_data.py --profile "${PROFILE}"
bash scripts/launch_sft.sh "${PROFILE}"

