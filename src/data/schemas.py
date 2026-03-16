from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

from src.data.constants import CANONICAL_CATEGORIES, REQUIRED_DPO_FIELDS, REQUIRED_SFT_FIELDS


class SFTRecord(TypedDict, total=False):
    id: str
    category: str
    system: str
    instruction: str
    input: str
    output: str
    source: str
    source_id: str


class PreferenceRecord(TypedDict, total=False):
    id: str
    category: str
    prompt: str
    chosen: str
    rejected: str
    source: str
    source_id: str


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    reason: str


def issues_to_strings(issues: list[ValidationIssue]) -> list[str]:
    return [f"{issue.field}: {issue.reason}" for issue in issues]


def _validate_required_text(
    record: Mapping[str, Any], fields: list[str], allow_empty: set[str] | None = None
) -> list[ValidationIssue]:
    allow_empty = allow_empty or set()
    issues: list[ValidationIssue] = []
    for field in fields:
        value = record.get(field)
        if value is None:
            issues.append(ValidationIssue(field=field, reason="missing"))
            continue
        if not isinstance(value, str):
            issues.append(ValidationIssue(field=field, reason="must be string"))
            continue
        if field not in allow_empty and not value.strip():
            issues.append(ValidationIssue(field=field, reason="must be non-empty"))
    return issues


def validate_sft_record(record: Mapping[str, Any]) -> list[ValidationIssue]:
    issues = _validate_required_text(record, REQUIRED_SFT_FIELDS, allow_empty={"input"})
    category = record.get("category", "")
    if isinstance(category, str) and category and category not in CANONICAL_CATEGORIES:
        issues.append(ValidationIssue(field="category", reason=f"unsupported category: {category}"))
    for optional_field in ["source", "source_id"]:
        value = record.get(optional_field)
        if value is not None and not isinstance(value, str):
            issues.append(ValidationIssue(field=optional_field, reason="must be string when provided"))
    return issues


def validate_preference_record(record: Mapping[str, Any]) -> list[ValidationIssue]:
    issues = _validate_required_text(record, REQUIRED_DPO_FIELDS)
    category = record.get("category", "")
    if isinstance(category, str) and category and category not in CANONICAL_CATEGORIES:
        issues.append(ValidationIssue(field="category", reason=f"unsupported category: {category}"))
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if isinstance(chosen, str) and isinstance(rejected, str) and chosen.strip() == rejected.strip():
        issues.append(ValidationIssue(field="chosen/rejected", reason="chosen and rejected must differ"))
    for optional_field in ["source", "source_id"]:
        value = record.get(optional_field)
        if value is not None and not isinstance(value, str):
            issues.append(ValidationIssue(field=optional_field, reason="must be string when provided"))
    return issues
