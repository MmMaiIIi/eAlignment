from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.comparison import (  # noqa: E402
    ComparisonThresholds,
    align_prediction_rows,
    build_markdown_report,
    evaluate_comparison_rows,
    summarize_results,
)
from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.jsonl import read_jsonl, write_jsonl  # noqa: E402
from src.utils.paths import from_root  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 evaluation and badcase collection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=from_root("configs", "eval", "comparison_eval.yaml"),
        help="Evaluation config file for thresholds and default paths.",
    )
    parser.add_argument("--base-predictions", type=Path, default=None)
    parser.add_argument("--sft-predictions", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=from_root("reports", "experiments", "latest_eval"),
        help="Directory for summary.json, per_sample.jsonl, badcases.jsonl, report.md",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    base_path = args.base_predictions
    sft_path = args.sft_predictions
    if base_path is None and cfg.get("base_predictions"):
        base_path = from_root(str(cfg["base_predictions"]))
    if sft_path is None and cfg.get("sft_predictions"):
        sft_path = from_root(str(cfg["sft_predictions"]))

    if sft_path is None:
        raise ValueError("sft predictions path is required via --sft-predictions or config.")

    base_rows = read_jsonl(base_path) if base_path is not None and base_path.exists() else None
    sft_rows = read_jsonl(sft_path)

    thresholds = ComparisonThresholds(
        low_score_threshold=float(cfg.get("low_score_threshold", 0.6)),
        regression_threshold=float(cfg.get("regression_threshold", 0.15)),
    )
    aligned = align_prediction_rows(base_rows=base_rows, sft_rows=sft_rows)
    mode = "comparison" if base_rows is not None else "single_sft"
    per_sample, badcases = evaluate_comparison_rows(aligned_rows=aligned, thresholds=thresholds)
    summary = summarize_results(per_sample=per_sample, thresholds=thresholds, mode=mode)
    report_md = build_markdown_report(summary=summary, per_sample=per_sample)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_jsonl(output_dir / "badcases.jsonl", badcases)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Mode: {mode}")
    print(f"Samples evaluated: {summary['counts']['samples']}")
    print(f"Badcases: {summary['counts']['badcases']}")
    print(f"Summary -> {output_dir / 'summary.json'}")
    print(f"Per-sample -> {output_dir / 'per_sample.jsonl'}")
    print(f"Badcases -> {output_dir / 'badcases.jsonl'}")
    print(f"Markdown report -> {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
