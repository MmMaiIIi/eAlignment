from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pipeline import process_preference_raw_records
from src.utils.config import load_yaml_config
from src.utils.jsonl import read_jsonl, write_jsonl
from src.utils.paths import from_root


def write_dataset_info(path: Path) -> None:
    content = {
        "ecom_sft_seed": {
            "file_name": "sft_train.jsonl",
            "columns": {
                "instruction": "instruction",
                "input": "input",
                "output": "output",
            },
        },
        "ecom_pref_seed": {
            "file_name": "dpo_train.jsonl",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def prepare(
    input_path: Path,
    output_dir: Path,
    rejected_output: Path,
    split_config_path: Path,
    source_name: str,
    fail_on_invalid: bool,
) -> dict[str, Any]:
    split_cfg = load_yaml_config(split_config_path)
    train_ratio = float(split_cfg["train_ratio"])
    dev_ratio = float(split_cfg["dev_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    seed = int(split_cfg["seed"])

    raw_records = read_jsonl(input_path)
    splits, valid_records, rejected_records = process_preference_raw_records(
        raw_records=raw_records,
        source_name=source_name,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if fail_on_invalid and rejected_records:
        raise ValueError(f"Found {len(rejected_records)} invalid records in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "dpo_all.jsonl", valid_records)
    write_jsonl(output_dir / "dpo_train.jsonl", splits["train"])
    write_jsonl(output_dir / "dpo_dev.jsonl", splits["dev"])
    write_jsonl(output_dir / "dpo_test.jsonl", splits["test"])
    write_jsonl(rejected_output, rejected_records)

    summary = {
        "dataset": "dpo",
        "input": str(input_path),
        "total_raw": len(raw_records),
        "valid": len(valid_records),
        "rejected": len(rejected_records),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare preference JSONL for LLaMA-Factory")
    parser.add_argument("--input", type=Path, default=from_root("data", "raw", "mock_dpo_raw.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=from_root("data", "processed"))
    parser.add_argument(
        "--rejected-output", type=Path, default=from_root("data", "interim", "dpo_rejected.jsonl")
    )
    parser.add_argument(
        "--split-config", type=Path, default=from_root("configs", "data", "split.yaml")
    )
    parser.add_argument("--source-name", type=str, default="mock_dpo_raw")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument(
        "--dataset-info",
        type=Path,
        default=from_root("data", "processed", "dataset_info.json"),
        help="Path to write dataset_info.json consumed by LLaMA-Factory",
    )
    args = parser.parse_args()

    summary = prepare(
        input_path=args.input,
        output_dir=args.output_dir,
        rejected_output=args.rejected_output,
        split_config_path=args.split_config,
        source_name=args.source_name,
        fail_on_invalid=args.fail_on_invalid,
    )
    write_dataset_info(args.dataset_info)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote dataset info -> {args.dataset_info}")


if __name__ == "__main__":
    main()
