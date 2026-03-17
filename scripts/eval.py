from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.config import load_profile, load_yaml, resolve
from align.eval import run_eval_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation and comparison.")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Override eval config path")
    parser.add_argument("--base", type=Path, default=None, help="Override base predictions JSONL")
    parser.add_argument("--tuned", type=Path, default=None, help="Override tuned predictions JSONL")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    profile_name, profile = load_profile(args.profile)
    eval_cfg_path = args.config or resolve(profile.get("eval_config", "configs/eval.yaml"))
    eval_cfg = load_yaml(eval_cfg_path)

    raw_base = eval_cfg.get("base_predictions")
    base = args.base if args.base is not None else (resolve(raw_base) if raw_base else None)
    tuned = args.tuned if args.tuned is not None else resolve(eval_cfg["tuned_predictions"])
    output_dir = args.output_dir or resolve(eval_cfg.get("output_dir", profile.get("eval_output_dir", "reports/experiments/latest_eval")))

    summary = run_eval_pipeline(
        base_path=base,
        tuned_path=tuned,
        output_dir=output_dir,
        low_score_threshold=float(eval_cfg.get("low_score_threshold", 0.6)),
        regression_threshold=float(eval_cfg.get("regression_threshold", 0.15)),
    )
    print(json.dumps({"profile": profile_name, "output_dir": str(output_dir), **summary["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
