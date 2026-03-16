from src.utils.jsonl import read_jsonl
from src.utils.paths import from_root


def test_sft_seed_readable() -> None:
    rows = read_jsonl(from_root("data", "synthetic", "sft_seed.jsonl"))
    assert rows
    first = rows[0]
    for key in ["id", "category", "system", "instruction", "input", "output"]:
        assert key in first


def test_dpo_seed_readable() -> None:
    rows = read_jsonl(from_root("data", "synthetic", "dpo_seed.jsonl"))
    assert rows
    first = rows[0]
    for key in ["id", "category", "prompt", "chosen", "rejected"]:
        assert key in first


def test_raw_mock_assets_readable() -> None:
    sft_rows = read_jsonl(from_root("data", "raw", "mock_sft_raw.jsonl"))
    dpo_rows = read_jsonl(from_root("data", "raw", "mock_dpo_raw.jsonl"))
    assert len(sft_rows) >= 1
    assert len(dpo_rows) >= 1
