from __future__ import annotations

from typing import Any

from src.data.constants import CATEGORY_ALIASES


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_text(value: Any) -> str:
    return normalize_whitespace(to_text(value))


def normalize_optional_text(value: Any) -> str:
    text = to_text(value).strip()
    return " ".join(text.split()) if text else ""


def normalize_category(value: Any) -> str:
    token = normalize_text(value).lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(token, "")
