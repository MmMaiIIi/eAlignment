from src.data.normalization import normalize_category
from src.data.parsers import parse_raw_preference_record, parse_raw_sft_record
from src.data.schemas import validate_preference_record, validate_sft_record


def test_category_normalization_aliases() -> None:
    assert normalize_category("returns-refunds") == "returns_refunds"
    assert normalize_category("shipping") == "shipping_logistics"
    assert normalize_category("order change") == "order_modification"
    assert normalize_category("unknown_category") == ""


def test_sft_schema_validation_rejects_missing_required_field() -> None:
    record = {
        "id": "sft_bad_001",
        "category": "returns_refunds",
        "system": "You are support.",
        "instruction": "Customer asks for return.",
        "input": "",
        "output": "",
    }
    issues = validate_sft_record(record)
    assert any(issue.field == "output" for issue in issues)


def test_preference_schema_validation_rejects_identical_pairs() -> None:
    record = {
        "id": "pref_bad_001",
        "category": "complaint_soothing",
        "prompt": "Customer is upset about delay.",
        "chosen": "Please wait.",
        "rejected": "Please wait.",
    }
    issues = validate_preference_record(record)
    assert any(issue.field == "chosen/rejected" for issue in issues)


def test_parser_filters_malformed_records() -> None:
    bad_sft_raw = {"id": "bad_sft_001", "category": "invalid", "query": "hello", "response": "hi"}
    parsed_sft, sft_errors = parse_raw_sft_record(bad_sft_raw, line_no=1, source_name="test")
    assert parsed_sft is None
    assert sft_errors

    bad_dpo_raw = {
        "id": "bad_pref_001",
        "category": "shipping",
        "prompt": "where is order",
        "chosen": "same",
        "rejected": "same",
    }
    parsed_dpo, dpo_errors = parse_raw_preference_record(bad_dpo_raw, line_no=1, source_name="test")
    assert parsed_dpo is None
    assert dpo_errors
