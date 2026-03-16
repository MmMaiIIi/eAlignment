import subprocess
import sys
from pathlib import Path

from src.utils.jsonl import read_jsonl


def test_run_eval_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "eval.jsonl"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--predictions",
        "data/synthetic/eval_predictions.jsonl",
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    rows = read_jsonl(output_path)
    assert rows
    assert "proxy" in rows[0]
    assert "is_badcase" in rows[0]
