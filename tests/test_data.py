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
            "--dataset-info",
            str(dataset_info),
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


def test_prepare_data_external_source_formats(tmp_path: Path) -> None:
    samples = {
        "jddc": {
            "session_id": "jd_001",
            "dialog": [
                {"role": "user", "text": "Where is my order tracking update?"},
                {"role": "assistant", "text": "Please share your order number and I will check shipping status."},
            ],
        },
        "ecd": {
            "id": "ecd_001",
            "buyer_query": "I want to return this damaged item.",
            "seller_response": "Please share your order number and photos for return processing.",
            "context": "Order was delivered yesterday.",
        },
        "faq": {
            "faq_id": "faq_001",
            "question": "Is this bottle BPA free?",
            "answer": "Yes, the bottle material is BPA free.",
        },
    }
    required_keys = {"id", "category", "system", "instruction", "input", "output", "source", "source_id"}

    for source_format, row in samples.items():
        raw_path = tmp_path / f"{source_format}.jsonl"
        raw_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        out_dir = tmp_path / source_format / "processed"
        rejected_path = tmp_path / source_format / "interim" / "sft_rejected.jsonl"
        dataset_info = out_dir / "dataset_info.json"

        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_data.py",
                "--profile",
                "smoke",
                "--input",
                str(raw_path),
                "--output-dir",
                str(out_dir),
                "--rejected-path",
                str(rejected_path),
                "--dataset-info",
                str(dataset_info),
                "--source-format",
                source_format,
                "--fail-on-invalid",
            ],
            check=True,
        )

        sft_all = read_jsonl(out_dir / "sft_all.jsonl")
        sft_train = read_jsonl(out_dir / "sft_train.jsonl")
        rejected = read_jsonl(rejected_path)
        mapping = json.loads(dataset_info.read_text(encoding="utf-8"))

        assert len(sft_all) == 1
        assert len(sft_train) == 1
        assert len(rejected) == 0
        assert required_keys.issubset(sft_all[0].keys())
        assert sft_all[0]["category"]
        assert sft_all[0]["instruction"]
        assert sft_all[0]["output"]
        assert mapping["ecom_sft_seed"]["columns"] == {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        }


def test_prepare_data_external_format_reports_normalization_errors(tmp_path: Path) -> None:
    raw_path = tmp_path / "faq_bad.jsonl"
    raw_path.write_text(json.dumps({"faq_id": "faq_bad_001", "question": "How to return?"}) + "\n", encoding="utf-8")

    out_dir = tmp_path / "processed"
    rejected_path = tmp_path / "interim" / "sft_rejected.jsonl"
    dataset_info = out_dir / "dataset_info.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--profile",
            "smoke",
            "--input",
            str(raw_path),
            "--output-dir",
            str(out_dir),
            "--rejected-path",
            str(rejected_path),
            "--dataset-info",
            str(dataset_info),
            "--source-format",
            "faq",
        ],
        check=True,
    )

    rejected = read_jsonl(rejected_path)
    assert len(rejected) == 1
    assert any("normalize: faq: missing response text" in err for err in rejected[0]["errors"])
