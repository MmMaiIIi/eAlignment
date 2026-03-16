from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.proxy_rules import score_response
from src.utils.config import load_yaml_config
from src.utils.jsonl import read_jsonl, write_jsonl
from src.utils.paths import from_root


def evaluate(predictions: List[Dict[str, str]], threshold: float) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for record in predictions:
        proxy = score_response(record.get("response", ""))
        result = {
            "id": record.get("id", ""),
            "category": record.get("category", ""),
            "prompt": record.get("prompt", ""),
            "response": record.get("response", ""),
            "proxy": proxy,
            "is_badcase": proxy["score"] < threshold,
        }
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight proxy evaluation.")
    parser.add_argument(
        "--config", type=Path, default=from_root("configs", "eval", "proxy_eval.yaml")
    )
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    threshold = float(config.get("badcase_threshold", 0.67))
    predictions_path = (
        args.predictions if args.predictions is not None else from_root(config["input_predictions"])
    )
    output_path = args.output if args.output is not None else from_root(config["output_results"])

    predictions = read_jsonl(predictions_path)
    results = evaluate(predictions, threshold)
    write_jsonl(output_path, results)

    avg_score = sum(float(row["proxy"]["score"]) for row in results) / max(len(results), 1)
    badcases = sum(bool(row["is_badcase"]) for row in results)
    print(f"Evaluated {len(results)} records")
    print(f"Average proxy score: {avg_score:.4f}")
    print(f"Badcases: {badcases}")
    print(f"Saved results -> {output_path}")


if __name__ == "__main__":
    main()
