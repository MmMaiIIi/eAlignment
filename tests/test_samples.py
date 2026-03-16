from src.data.io import read_jsonl


def test_sft_sample_jsonl_readable() -> None:
    rows = read_jsonl("data/samples/sft_sample.jsonl")
    assert len(rows) >= 2
    assert {"instruction", "input", "output"}.issubset(rows[0].keys())


def test_dpo_sample_jsonl_readable() -> None:
    rows = read_jsonl("data/samples/dpo_sample.jsonl")
    assert len(rows) >= 2
    assert {"prompt", "chosen", "rejected"}.issubset(rows[0].keys())

