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


def test_eval_comparison_assets_readable() -> None:
    base_rows = read_jsonl(from_root("data", "synthetic", "eval_base_predictions.jsonl"))
    sft_rows = read_jsonl(from_root("data", "synthetic", "eval_sft_predictions.jsonl"))
    assert len(base_rows) == len(sft_rows)
    assert len(base_rows) >= 1


def test_reporting_templates_exist() -> None:
    required = [
        from_root("reports", "badcases", "WORKFLOW.md"),
        from_root("reports", "templates", "badcase_analysis_template.md"),
        from_root("reports", "templates", "ablation_base_vs_sft.md"),
        from_root("reports", "templates", "ablation_sft_vs_sft_dpo.md"),
        from_root("reports", "templates", "ablation_lora_vs_qlora.md"),
        from_root("reports", "templates", "ablation_dpo_beta_comparison.md"),
        from_root("reports", "templates", "ablation_preference_quality_sensitivity.md"),
    ]
    for path in required:
        assert path.exists()
