from __future__ import annotations

from typing import Any, Mapping

from src.data.normalization import normalize_category, normalize_optional_text, normalize_text
from src.data.schemas import (
    PreferenceRecord,
    SFTRecord,
    issues_to_strings,
    validate_preference_record,
    validate_sft_record,
)

DEFAULT_SYSTEM_PROMPT = "You are a professional e-commerce customer support assistant."

SFT_KEY_MAP = {
    "id": ["id", "sample_id", "record_id", "uid"],
    "category": ["category", "intent", "topic"],
    "system": ["system", "system_prompt", "sys_prompt"],
    "instruction": ["instruction", "query", "customer_query", "prompt", "user_message"],
    "input": ["input", "context", "extra_context"],
    "output": ["output", "response", "assistant_response", "answer"],
    "source": ["source", "source_name"],
    "source_id": ["source_id", "original_id", "ticket_id"],
}

DPO_KEY_MAP = {
    "id": ["id", "sample_id", "record_id", "uid"],
    "category": ["category", "intent", "topic"],
    "prompt": ["prompt", "instruction", "query", "user_message"],
    "chosen": ["chosen", "preferred", "good_response"],
    "rejected": ["rejected", "dispreferred", "bad_response"],
    "source": ["source", "source_name"],
    "source_id": ["source_id", "original_id", "ticket_id"],
}


def _pick_value(raw: Mapping[str, Any], aliases: list[str], default: Any = None) -> Any:
    for key in aliases:
        if key in raw:
            return raw[key]
    return default


def parse_raw_sft_record(
    raw: Mapping[str, Any],
    line_no: int,
    source_name: str,
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[SFTRecord | None, list[str]]:
    record: SFTRecord = {
        "id": normalize_text(_pick_value(raw, SFT_KEY_MAP["id"])),
        "category": normalize_category(_pick_value(raw, SFT_KEY_MAP["category"])),
        "system": normalize_text(_pick_value(raw, SFT_KEY_MAP["system"], default_system_prompt)),
        "instruction": normalize_text(_pick_value(raw, SFT_KEY_MAP["instruction"])),
        "input": normalize_optional_text(_pick_value(raw, SFT_KEY_MAP["input"], "")),
        "output": normalize_text(_pick_value(raw, SFT_KEY_MAP["output"])),
        "source": normalize_optional_text(_pick_value(raw, SFT_KEY_MAP["source"], source_name)),
        "source_id": normalize_optional_text(_pick_value(raw, SFT_KEY_MAP["source_id"], "")),
    }
    if not record["source"]:
        record["source"] = source_name
    issues = validate_sft_record(record)
    if issues:
        return None, issues_to_strings(issues)
    return record, []


def parse_raw_preference_record(
    raw: Mapping[str, Any], line_no: int, source_name: str
) -> tuple[PreferenceRecord | None, list[str]]:
    record: PreferenceRecord = {
        "id": normalize_text(_pick_value(raw, DPO_KEY_MAP["id"])),
        "category": normalize_category(_pick_value(raw, DPO_KEY_MAP["category"])),
        "prompt": normalize_text(_pick_value(raw, DPO_KEY_MAP["prompt"])),
        "chosen": normalize_text(_pick_value(raw, DPO_KEY_MAP["chosen"])),
        "rejected": normalize_text(_pick_value(raw, DPO_KEY_MAP["rejected"])),
        "source": normalize_optional_text(_pick_value(raw, DPO_KEY_MAP["source"], source_name)),
        "source_id": normalize_optional_text(_pick_value(raw, DPO_KEY_MAP["source_id"], "")),
    }
    if not record["source"]:
        record["source"] = source_name
    issues = validate_preference_record(record)
    if issues:
        return None, issues_to_strings(issues)
    return record, []


def build_rejected_record(
    raw: Mapping[str, Any], line_no: int, errors: list[str], dataset: str, source_name: str
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "line_no": line_no,
        "source": source_name,
        "raw": dict(raw),
        "errors": errors,
    }
