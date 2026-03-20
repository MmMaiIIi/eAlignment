import json
import subprocess
import sys
from pathlib import Path

import pytest

from align.config import resolve
from align.io import read_jsonl


def test_sft_eval_prompt_file_is_valid() -> None:
    prompt_path = resolve("data/synthetic/sft_eval_prompts.jsonl")
    rows = read_jsonl(prompt_path)
    assert len(rows) >= 3
    for row in rows:
        assert isinstance(row.get("id"), str)
        assert isinstance(row.get("prompt"), str)
        assert row["prompt"].strip()


def test_plot_loss_script_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    training_log = tmp_path / "training_log.jsonl"
    training_log.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 2.1}),
                json.dumps({"step": 2, "loss": 1.8}),
                json.dumps({"step": 2, "eval_loss": 1.9}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_png = tmp_path / "loss_curve.png"
    output_json = tmp_path / "loss_curve.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_loss.py",
            "--run-name",
            "test_run",
            "--training-log",
            str(training_log),
            "--output-png",
            str(output_png),
            "--output-json",
            str(output_json),
        ],
        check=True,
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_png.exists()
    assert summary["run_name"] == "test_run"
    assert summary["counts"]["train_points"] == 2
    assert summary["counts"]["eval_points"] == 1
