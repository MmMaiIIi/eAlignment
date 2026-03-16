import json
from pathlib import Path

from src.utils.config import load_yaml_config


def test_yaml_configs_load() -> None:
    config_paths = [
        Path("configs/data/sft_schema.yaml"),
        Path("configs/data/preference_schema.yaml"),
        Path("configs/data/split.yaml"),
        Path("configs/eval/proxy_eval.yaml"),
        Path("configs/llamafactory/sft/qwen3_8b_lora_sft.yaml"),
        Path("configs/llamafactory/dpo/qwen3_8b_lora_dpo.yaml"),
    ]
    for path in config_paths:
        data = load_yaml_config(path)
        assert isinstance(data, dict)
        assert len(data) > 0


def test_deepspeed_json_loads() -> None:
    path = Path("configs/llamafactory/deepspeed/zero2.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["zero_optimization"]["stage"] == 2
