import json
import subprocess
import sys
from pathlib import Path

from align.io import read_jsonl


def test_eval_pipeline_and_badcases_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    badcase_md = tmp_path / "badcases.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--profile",
            "eval",
            "--base",
            "data/synthetic/eval_base_predictions.jsonl",
            "--tuned",
            "data/synthetic/eval_sft_predictions.jsonl",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/badcases.py",
            "--profile",
            "eval",
            "--input",
            str(output_dir / "badcases.jsonl"),
            "--output",
            str(badcase_md),
            "--top-k",
            "5",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    per_sample = read_jsonl(output_dir / "per_sample.jsonl")
    badcases = read_jsonl(output_dir / "badcases.jsonl")

    assert summary["counts"]["samples"] >= 1
    assert len(per_sample) >= 1
    assert isinstance(badcases, list)
    assert badcase_md.exists()


def test_ablation_plan_smoke(tmp_path: Path) -> None:
    out_md = tmp_path / "ablation_plan.md"
    out_json = tmp_path / "ablation_plan.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/plan_ablations.py",
            "--config",
            "configs/ablations.yaml",
            "--output-md",
            str(out_md),
            "--output-json",
            str(out_json),
            "--check-paths",
        ],
        check=True,
    )
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["name"] == "stage5_ablation_matrix"
    assert len(plan["experiments"]) >= 5

