from pathlib import Path

import pytest

from src.utils.config_loader import load_config


@pytest.mark.parametrize(
    "config_path,expected_stage",
    [
        ("configs/sft_smoke.yaml", "sft"),
        ("configs/dpo_smoke.yaml", "dpo"),
        ("configs/eval_smoke.yaml", "eval"),
    ],
)
def test_config_files_load(config_path: str, expected_stage: str) -> None:
    cfg = load_config(Path(config_path))
    assert cfg["stage"] == expected_stage
    assert "model" in cfg
    assert "data" in cfg

