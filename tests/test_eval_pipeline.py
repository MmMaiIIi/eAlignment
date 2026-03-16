import json
import subprocess
import sys
from pathlib import Path

from src.utils.jsonl import read_jsonl


def test_run_eval_comparison_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval_out"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--config",
        "configs/eval/comparison_eval.yaml",
        "--base-predictions",
        "data/synthetic/eval_base_predictions.jsonl",
        "--sft-predictions",
        "data/synthetic/eval_sft_predictions.jsonl",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)

    summary_path = output_dir / "summary.json"
    per_sample_path = output_dir / "per_sample.jsonl"
    badcases_path = output_dir / "badcases.jsonl"
    report_path = output_dir / "report.md"

    assert summary_path.exists()
    assert per_sample_path.exists()
    assert badcases_path.exists()
    assert report_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_sample = read_jsonl(per_sample_path)

    assert summary["mode"] == "comparison"
    assert summary["counts"]["samples"] >= 1
    assert "category_breakdown" in summary
    assert per_sample
    assert "sft" in per_sample[0]
    assert "badcase_reasons" in per_sample[0]


def test_summarize_badcases_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval_out"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--config",
            "configs/eval/comparison_eval.yaml",
            "--base-predictions",
            "data/synthetic/eval_base_predictions.jsonl",
            "--sft-predictions",
            "data/synthetic/eval_sft_predictions.jsonl",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    summary_md = tmp_path / "badcases.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_badcases.py",
            "--badcase-file",
            str(output_dir / "badcases.jsonl"),
            "--output-md",
            str(summary_md),
            "--top-k",
            "3",
        ],
        check=True,
    )
    assert summary_md.exists()
    text = summary_md.read_text(encoding="utf-8")
    assert "Badcase Summary" in text
