import subprocess
import sys
from pathlib import Path

from src.utils.jsonl import read_jsonl


def test_prepare_sft_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    rejected_path = tmp_path / "interim" / "sft_rejected.jsonl"
    cmd = [
        sys.executable,
        "scripts/prepare_sft_data.py",
        "--input",
        "data/raw/mock_sft_raw.jsonl",
        "--output-dir",
        str(output_dir),
        "--rejected-output",
        str(rejected_path),
        "--split-config",
        "configs/data/split.yaml",
    ]
    subprocess.run(cmd, check=True)
    all_rows = read_jsonl(output_dir / "sft_all.jsonl")
    train_rows = read_jsonl(output_dir / "sft_train.jsonl")
    dev_rows = read_jsonl(output_dir / "sft_dev.jsonl")
    test_rows = read_jsonl(output_dir / "sft_test.jsonl")
    rejected_rows = read_jsonl(rejected_path)

    assert len(all_rows) == 6
    assert len(rejected_rows) == 2
    assert len(train_rows) + len(dev_rows) + len(test_rows) == len(all_rows)
    assert all("category" in row and "source" in row for row in all_rows)


def test_prepare_dpo_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    rejected_path = tmp_path / "interim" / "dpo_rejected.jsonl"
    info_path = output_dir / "dataset_info.json"
    cmd = [
        sys.executable,
        "scripts/prepare_dpo_data.py",
        "--input",
        "data/raw/mock_dpo_raw.jsonl",
        "--output-dir",
        str(output_dir),
        "--rejected-output",
        str(rejected_path),
        "--split-config",
        "configs/data/split.yaml",
        "--dataset-info",
        str(info_path),
    ]
    subprocess.run(cmd, check=True)
    all_rows = read_jsonl(output_dir / "dpo_all.jsonl")
    train_rows = read_jsonl(output_dir / "dpo_train.jsonl")
    dev_rows = read_jsonl(output_dir / "dpo_dev.jsonl")
    test_rows = read_jsonl(output_dir / "dpo_test.jsonl")
    rejected_rows = read_jsonl(rejected_path)

    assert len(all_rows) == 4
    assert len(rejected_rows) == 2
    assert len(train_rows) + len(dev_rows) + len(test_rows) == len(all_rows)
    assert all("category" in row and "source" in row for row in all_rows)
    assert info_path.exists()
