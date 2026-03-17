import json
from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.llamafactory import missing_required_dpo_fields, missing_required_sft_fields

SFT_CONFIGS = [
    Path("configs/llamafactory/sft/smoke.yaml"),
    Path("configs/llamafactory/sft/qwen3_8b_lora.yaml"),
    Path("configs/llamafactory/sft/qwen3_8b_qlora.yaml"),
]

OTHER_CONFIGS = [
    Path("configs/data/sft_schema.yaml"),
    Path("configs/data/preference_schema.yaml"),
    Path("configs/data/split.yaml"),
    Path("configs/eval/proxy_eval.yaml"),
    Path("configs/eval/comparison_eval.yaml"),
    Path("configs/llamafactory/dpo/qwen3_8b_lora_dpo.yaml"),
    Path("configs/llamafactory/dpo/smoke.yaml"),
    Path("configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml"),
]


def test_yaml_configs_load() -> None:
    config_paths = OTHER_CONFIGS + SFT_CONFIGS
    for path in config_paths:
        data = load_yaml_config(path)
        assert isinstance(data, dict)
        assert len(data) > 0


def test_deepspeed_json_loads() -> None:
    path = Path("configs/llamafactory/deepspeed/zero2.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["zero_optimization"]["stage"] == 2


def test_sft_configs_have_required_fields() -> None:
    for config_path in SFT_CONFIGS:
        config = load_yaml_config(config_path)
        missing = missing_required_sft_fields(config)
        assert not missing, f"{config_path} missing fields: {missing}"
        assert config["stage"] == "sft"


def test_lora_and_qlora_specific_fields() -> None:
    lora = load_yaml_config(Path("configs/llamafactory/sft/qwen3_8b_lora.yaml"))
    qlora = load_yaml_config(Path("configs/llamafactory/sft/qwen3_8b_qlora.yaml"))
    assert lora["finetuning_type"] == "lora"
    assert qlora["finetuning_type"] == "lora"
    assert int(qlora["quantization_bit"]) == 4


def test_eval_comparison_config_fields() -> None:
    cfg = load_yaml_config(Path("configs/eval/comparison_eval.yaml"))
    for key in ["base_predictions", "sft_predictions", "low_score_threshold", "regression_threshold"]:
        assert key in cfg


def test_dpo_configs_have_required_fields() -> None:
    for path in [
        Path("configs/llamafactory/dpo/smoke.yaml"),
        Path("configs/llamafactory/dpo/qwen3_8b_dpo_lora.yaml"),
    ]:
        cfg = load_yaml_config(path)
        missing = missing_required_dpo_fields(cfg)
        assert not missing, f"{path} missing fields: {missing}"
        assert cfg["stage"] == "dpo"
