from __future__ import annotations

from typing import Any, Mapping

from src.data.parsers import (
    build_rejected_record,
    parse_raw_preference_record,
    parse_raw_sft_record,
)
from src.data.splitting import split_records


def process_sft_raw_records(
    raw_records: list[Mapping[str, Any]],
    source_name: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_records, start=1):
        record, errors = parse_raw_sft_record(raw=raw, line_no=line_no, source_name=source_name)
        if errors:
            rejected.append(
                build_rejected_record(
                    raw=raw, line_no=line_no, errors=errors, dataset="sft", source_name=source_name
                )
            )
            continue
        valid.append(record)  # type: ignore[arg-type]
    splits = split_records(
        valid, train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio, seed=seed
    )
    return splits, valid, rejected


def process_preference_raw_records(
    raw_records: list[Mapping[str, Any]],
    source_name: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_records, start=1):
        record, errors = parse_raw_preference_record(raw=raw, line_no=line_no, source_name=source_name)
        if errors:
            rejected.append(
                build_rejected_record(
                    raw=raw, line_no=line_no, errors=errors, dataset="dpo", source_name=source_name
                )
            )
            continue
        valid.append(record)  # type: ignore[arg-type]
    splits = split_records(
        valid, train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio, seed=seed
    )
    return splits, valid, rejected
