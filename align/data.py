from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
from typing import Any, Mapping

from align.io import read_jsonl, write_json, write_jsonl
from align.prompts import SYSTEM_PROMPT


CATEGORIES = [
    "returns_refunds",
    "shipping_logistics",
    "product_specs",
    "order_modification",
    "after_sales",
    "complaint_soothing",
]

CATEGORY_ALIASES = {
    "returns_refunds": "returns_refunds",
    "returns": "returns_refunds",
    "refund": "returns_refunds",
    "refunds": "returns_refunds",
    "return": "returns_refunds",
    "return_refund": "returns_refunds",
    "shipping_logistics": "shipping_logistics",
    "shipping": "shipping_logistics",
    "logistics": "shipping_logistics",
    "delivery": "shipping_logistics",
    "product_specs": "product_specs",
    "product_spec": "product_specs",
    "specs": "product_specs",
    "product": "product_specs",
    "order_modification": "order_modification",
    "order_change": "order_modification",
    "change_order": "order_modification",
    "modify_order": "order_modification",
    "after_sales": "after_sales",
    "aftersales": "after_sales",
    "post_sale": "after_sales",
    "warranty": "after_sales",
    "complaint_soothing": "complaint_soothing",
    "complaint": "complaint_soothing",
    "deescalation": "complaint_soothing",
}

SFT_KEYS = {
    "id": ["id", "sample_id", "record_id", "uid"],
    "category": ["category", "intent", "topic"],
    "system": ["system", "system_prompt", "sys_prompt"],
    "instruction": ["instruction", "query", "customer_query", "prompt", "user_message"],
    "input": ["input", "context", "extra_context"],
    "output": ["output", "response", "assistant_response", "answer"],
    "source": ["source", "source_name"],
    "source_id": ["source_id", "original_id", "ticket_id"],
}

PREF_KEYS = {
    "id": ["id", "sample_id", "record_id", "uid"],
    "category": ["category", "intent", "topic"],
    "prompt": ["prompt", "instruction", "query", "user_message"],
    "chosen": ["chosen", "preferred", "good_response"],
    "rejected": ["rejected", "dispreferred", "bad_response"],
    "source": ["source", "source_name"],
    "source_id": ["source_id", "original_id", "ticket_id"],
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _category(value: Any) -> str:
    token = _text(value).lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(token, "")


def _pick(raw: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _require_text(record: Mapping[str, Any], fields: list[str], allow_empty: set[str] | None = None) -> list[str]:
    allow_empty = allow_empty or set()
    errors: list[str] = []
    for field in fields:
        value = record.get(field)
        if not isinstance(value, str):
            errors.append(f"{field}: must be string")
            continue
        if field not in allow_empty and not value.strip():
            errors.append(f"{field}: must be non-empty")
    return errors


def _split(records: list[dict[str, Any]], train: float, dev: float, test: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    if abs((train + dev + test) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * train)
    n_dev = int(n * dev)
    n_test = n - n_train - n_dev
    if n >= 3 and n_dev == 0:
        n_dev = 1
        n_train = max(n_train - 1, 1)
        n_test = n - n_train - n_dev
    if n >= 3 and n_test == 0:
        n_test = 1
        n_train = max(n_train - 1, 1)
    return {
        "train": rows[:n_train],
        "dev": rows[n_train : n_train + n_dev],
        "test": rows[n_train + n_dev : n_train + n_dev + n_test],
    }


def _rejected(raw: Mapping[str, Any], line_no: int, errors: list[str], source_name: str, dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "line_no": line_no,
        "source": source_name,
        "errors": errors,
        "raw": dict(raw),
    }


def _parse_sft(raw: Mapping[str, Any], source_name: str) -> tuple[dict[str, Any], list[str]]:
    record = {
        "id": _text(_pick(raw, SFT_KEYS["id"])),
        "category": _category(_pick(raw, SFT_KEYS["category"])),
        "system": _text(_pick(raw, SFT_KEYS["system"], SYSTEM_PROMPT)),
        "instruction": _text(_pick(raw, SFT_KEYS["instruction"])),
        "input": _text(_pick(raw, SFT_KEYS["input"], "")),
        "output": _text(_pick(raw, SFT_KEYS["output"])),
        "source": _text(_pick(raw, SFT_KEYS["source"], source_name)) or source_name,
        "source_id": _text(_pick(raw, SFT_KEYS["source_id"], "")),
    }
    errors = _require_text(record, ["id", "category", "system", "instruction", "input", "output"], {"input"})
    if record["category"] not in CATEGORIES:
        errors.append(f"category: unsupported `{record['category']}`")
    return record, errors


def _parse_pref(raw: Mapping[str, Any], source_name: str) -> tuple[dict[str, Any], list[str]]:
    record = {
        "id": _text(_pick(raw, PREF_KEYS["id"])),
        "category": _category(_pick(raw, PREF_KEYS["category"])),
        "prompt": _text(_pick(raw, PREF_KEYS["prompt"])),
        "chosen": _text(_pick(raw, PREF_KEYS["chosen"])),
        "rejected": _text(_pick(raw, PREF_KEYS["rejected"])),
        "source": _text(_pick(raw, PREF_KEYS["source"], source_name)) or source_name,
        "source_id": _text(_pick(raw, PREF_KEYS["source_id"], "")),
    }
    errors = _require_text(record, ["id", "category", "prompt", "chosen", "rejected"])
    if record["category"] not in CATEGORIES:
        errors.append(f"category: unsupported `{record['category']}`")
    if record["chosen"].strip() == record["rejected"].strip():
        errors.append("chosen/rejected: chosen and rejected must differ")
    return record, errors


def _quality_report(raw_count: int, valid_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts = Counter(
        {"empty_chosen_or_rejected": 0, "chosen_identical_to_rejected": 0, "malformed_examples": 0}
    )
    for rejected in rejected_rows:
        errors = [str(err).lower() for err in rejected.get("errors", [])]
        issue_counts["malformed_examples"] += 1
        if any(("chosen" in err or "rejected" in err) and "non-empty" in err for err in errors):
            issue_counts["empty_chosen_or_rejected"] += 1
        if any("chosen/rejected" in err and "must differ" in err for err in errors):
            issue_counts["chosen_identical_to_rejected"] += 1

    categories = Counter(str(row.get("category", "unknown")) for row in valid_rows)
    max_count = max(categories.values()) if categories else 0
    min_count = min(categories.values()) if categories else 0
    imbalance = (max_count / min_count) if min_count > 0 else 0.0

    prompt_map: dict[str, list[str]] = {}
    pair_map: dict[str, list[str]] = {}
    for idx, row in enumerate(valid_rows):
        row_id = row.get("id") or f"pref_{idx:04d}"
        prompt = str(row.get("prompt", "")).strip().lower()
        pair = f"{prompt}|||{str(row.get('chosen', '')).strip().lower()}|||{str(row.get('rejected', '')).strip().lower()}"
        prompt_map.setdefault(prompt, []).append(str(row_id))
        pair_map.setdefault(pair, []).append(str(row_id))

    dup_prompts = [ids for key, ids in prompt_map.items() if key and len(ids) > 1]
    dup_pairs = [ids for key, ids in pair_map.items() if key and len(ids) > 1]

    return {
        "dataset": "dpo",
        "total_raw": raw_count,
        "valid": len(valid_rows),
        "rejected": len(rejected_rows),
        "valid_ratio": (len(valid_rows) / raw_count) if raw_count else 0.0,
        "issue_counts": dict(issue_counts),
        "category_distribution": dict(categories),
        "category_imbalance": {
            "max_count": max_count,
            "min_count": min_count,
            "imbalance_ratio": imbalance,
            "flagged": imbalance >= 2.5 if min_count > 0 else False,
        },
        "duplicate_patterns": {
            "duplicate_prompt_groups": len(dup_prompts),
            "duplicate_pair_groups": len(dup_pairs),
            "duplicate_prompt_examples": dup_prompts[:5],
            "duplicate_pair_examples": dup_pairs[:5],
        },
    }


def _write_dataset_info(path: Path) -> None:
    data = {
        "ecom_sft_seed": {
            "file_name": "sft_train.jsonl",
            "columns": {"instruction": "instruction", "input": "input", "output": "output"},
        },
        "ecom_pref_seed": {
            "file_name": "dpo_train.jsonl",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    write_json(path, data)


def prepare_sft_dataset(
    input_path: Path,
    output_dir: Path,
    rejected_path: Path,
    split_cfg: Mapping[str, Any],
    source_name: str,
    fail_on_invalid: bool = False,
) -> dict[str, Any]:
    raw_rows = read_jsonl(input_path)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_rows, start=1):
        parsed, errors = _parse_sft(raw, source_name=source_name)
        if errors:
            rejected_rows.append(_rejected(raw, line_no, errors, source_name, "sft"))
        else:
            valid_rows.append(parsed)

    if fail_on_invalid and rejected_rows:
        raise ValueError(f"Found {len(rejected_rows)} invalid SFT rows.")

    splits = _split(
        valid_rows,
        train=float(split_cfg["train"]),
        dev=float(split_cfg["dev"]),
        test=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    write_jsonl(output_dir / "sft_all.jsonl", valid_rows)
    write_jsonl(output_dir / "sft_train.jsonl", splits["train"])
    write_jsonl(output_dir / "sft_dev.jsonl", splits["dev"])
    write_jsonl(output_dir / "sft_test.jsonl", splits["test"])
    write_jsonl(rejected_path, rejected_rows)

    return {
        "dataset": "sft",
        "input": str(input_path),
        "total_raw": len(raw_rows),
        "valid": len(valid_rows),
        "rejected": len(rejected_rows),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
    }


def prepare_preference_dataset(
    input_path: Path,
    output_dir: Path,
    rejected_path: Path,
    quality_path: Path,
    dataset_info_path: Path,
    split_cfg: Mapping[str, Any],
    source_name: str,
    fail_on_invalid: bool = False,
) -> dict[str, Any]:
    raw_rows = read_jsonl(input_path)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_rows, start=1):
        parsed, errors = _parse_pref(raw, source_name=source_name)
        if errors:
            rejected_rows.append(_rejected(raw, line_no, errors, source_name, "dpo"))
        else:
            valid_rows.append(parsed)

    if fail_on_invalid and rejected_rows:
        raise ValueError(f"Found {len(rejected_rows)} invalid preference rows.")

    splits = _split(
        valid_rows,
        train=float(split_cfg["train"]),
        dev=float(split_cfg["dev"]),
        test=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    write_jsonl(output_dir / "dpo_all.jsonl", valid_rows)
    write_jsonl(output_dir / "dpo_train.jsonl", splits["train"])
    write_jsonl(output_dir / "dpo_dev.jsonl", splits["dev"])
    write_jsonl(output_dir / "dpo_test.jsonl", splits["test"])
    write_jsonl(rejected_path, rejected_rows)

    quality = _quality_report(len(raw_rows), valid_rows, rejected_rows)
    write_json(quality_path, quality)
    _write_dataset_info(dataset_info_path)

    return {
        "dataset": "dpo",
        "input": str(input_path),
        "total_raw": len(raw_rows),
        "valid": len(valid_rows),
        "rejected": len(rejected_rows),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
        "quality_report": str(quality_path),
    }

