from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schemas import validate_record_fields
from src.utils.config import load_yaml_config
from src.utils.jsonl import read_jsonl, write_jsonl
from src.utils.paths import from_root


def normalize_record(record: Dict[str, str]) -> Dict[str, str]:
    return {
        "id": str(record["id"]).strip(),
        "category": str(record["category"]).strip(),
        "system": str(record["system"]).strip(),
        "instruction": str(record["instruction"]).strip(),
        "input": str(record.get("input", "")).strip(),
        "output": str(record["output"]).strip(),
    }


def prepare(input_path: Path, output_path: Path, schema_path: Path) -> int:
    schema = load_yaml_config(schema_path)
    required = schema["required_fields"]
    allowed_categories = set(schema["allowed_categories"])

    raw_records = read_jsonl(input_path)
    normalized: List[Dict[str, str]] = []

    for record in raw_records:
        missing = validate_record_fields(record, required)
        if missing:
            raise ValueError(f"Record {record.get('id', '<unknown>')} missing fields: {missing}")
        normalized_record = normalize_record(record)
        if normalized_record["category"] not in allowed_categories:
            raise ValueError(
                f"Record {normalized_record['id']} has invalid category: {normalized_record['category']}"
            )
        normalized.append(normalized_record)

    write_jsonl(output_path, normalized)
    return len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL for LLaMA-Factory")
    parser.add_argument("--input", type=Path, default=from_root("data", "synthetic", "sft_seed.jsonl"))
    parser.add_argument("--output", type=Path, default=from_root("data", "processed", "sft_train.jsonl"))
    parser.add_argument(
        "--schema", type=Path, default=from_root("configs", "data", "sft_schema.yaml")
    )
    args = parser.parse_args()

    count = prepare(args.input, args.output, args.schema)
    print(f"Prepared {count} SFT records -> {args.output}")


if __name__ == "__main__":
    main()
