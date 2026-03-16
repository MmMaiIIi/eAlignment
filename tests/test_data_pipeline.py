import subprocess
import sys
from pathlib import Path

from src.utils.jsonl import read_jsonl


def test_prepare_sft_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "sft_train.jsonl"
    cmd = [
        sys.executable,
        "scripts/prepare_sft_data.py",
        "--input",
        "data/synthetic/sft_seed.jsonl",
        "--output",
        str(output_path),
        "--schema",
        "configs/data/sft_schema.yaml",
    ]
    subprocess.run(cmd, check=True)
    rows = read_jsonl(output_path)
    assert len(rows) >= 1


def test_prepare_dpo_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "dpo_train.jsonl"
    info_path = tmp_path / "dataset_info.json"
    cmd = [
        sys.executable,
        "scripts/prepare_dpo_data.py",
        "--input",
        "data/synthetic/dpo_seed.jsonl",
        "--output",
        str(output_path),
        "--schema",
        "configs/data/preference_schema.yaml",
        "--dataset-info",
        str(info_path),
    ]
    subprocess.run(cmd, check=True)
    rows = read_jsonl(output_path)
    assert len(rows) >= 1
    assert info_path.exists()
