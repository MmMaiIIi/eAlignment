import json
import subprocess
import sys
from pathlib import Path

from align.io import read_jsonl


def test_prepare_data_and_pref_smoke(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    sft_rejected = interim / "sft_rejected.jsonl"
    dpo_rejected = interim / "dpo_rejected.jsonl"
    dpo_quality = interim / "dpo_quality_report.json"
    dataset_info = processed / "dataset_info.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--profile",
            "smoke",
            "--output-dir",
            str(processed),
            "--rejected-path",
            str(sft_rejected),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_pref.py",
            "--profile",
            "smoke",
            "--output-dir",
            str(processed),
            "--rejected-path",
            str(dpo_rejected),
            "--quality-path",
            str(dpo_quality),
            "--dataset-info",
            str(dataset_info),
        ],
        check=True,
    )

    sft_train = read_jsonl(processed / "sft_train.jsonl")
    dpo_train = read_jsonl(processed / "dpo_train.jsonl")
    sft_rej = read_jsonl(sft_rejected)
    dpo_rej = read_jsonl(dpo_rejected)
    quality = json.loads(dpo_quality.read_text(encoding="utf-8"))

    assert len(sft_train) >= 1
    assert len(dpo_train) >= 1
    assert len(sft_rej) >= 1
    assert len(dpo_rej) >= 1
    assert quality["dataset"] == "dpo"
    assert "issue_counts" in quality
    assert dataset_info.exists()

