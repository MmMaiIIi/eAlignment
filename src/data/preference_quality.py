from __future__ import annotations

from collections import Counter
from typing import Any


def _safe_id(record: dict[str, Any], fallback: str) -> str:
    value = record.get("id", "")
    text = str(value).strip()
    return text if text else fallback


def analyze_preference_quality(
    raw_count: int, valid_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]
) -> dict[str, Any]:
    issue_counts = Counter(
        {
            "empty_chosen_or_rejected": 0,
            "chosen_identical_to_rejected": 0,
            "malformed_examples": 0,
        }
    )

    for rejected in rejected_records:
        errors = [str(item).lower() for item in rejected.get("errors", [])]
        issue_counts["malformed_examples"] += 1
        if any(("chosen" in err or "rejected" in err) and "non-empty" in err for err in errors):
            issue_counts["empty_chosen_or_rejected"] += 1
        if any("chosen/rejected" in err and "must differ" in err for err in errors):
            issue_counts["chosen_identical_to_rejected"] += 1

    category_counter = Counter(str(row.get("category", "unknown")) for row in valid_records)
    max_count = max(category_counter.values()) if category_counter else 0
    min_count = min(category_counter.values()) if category_counter else 0
    imbalance_ratio = (max_count / min_count) if min_count > 0 else 0.0

    prompt_map: dict[str, list[str]] = {}
    pair_map: dict[str, list[str]] = {}
    for idx, row in enumerate(valid_records):
        row_id = _safe_id(row, fallback=f"valid_{idx:04d}")
        prompt_key = str(row.get("prompt", "")).strip().lower()
        pair_key = (
            f"{prompt_key}|||{str(row.get('chosen', '')).strip().lower()}|||"
            f"{str(row.get('rejected', '')).strip().lower()}"
        )
        prompt_map.setdefault(prompt_key, []).append(row_id)
        pair_map.setdefault(pair_key, []).append(row_id)

    duplicate_prompts = {k: v for k, v in prompt_map.items() if k and len(v) > 1}
    duplicate_pairs = {k: v for k, v in pair_map.items() if k and len(v) > 1}

    quality_report = {
        "dataset": "dpo",
        "total_raw": raw_count,
        "valid": len(valid_records),
        "rejected": len(rejected_records),
        "valid_ratio": (len(valid_records) / raw_count) if raw_count else 0.0,
        "issue_counts": dict(issue_counts),
        "category_distribution": dict(category_counter),
        "category_imbalance": {
            "max_count": max_count,
            "min_count": min_count,
            "imbalance_ratio": imbalance_ratio,
            "flagged": imbalance_ratio >= 2.5 if min_count > 0 else False,
        },
        "duplicate_patterns": {
            "duplicate_prompt_groups": len(duplicate_prompts),
            "duplicate_pair_groups": len(duplicate_pairs),
            "duplicate_prompt_examples": list(duplicate_prompts.values())[:5],
            "duplicate_pair_examples": list(duplicate_pairs.values())[:5],
        },
    }
    return quality_report
