from pathlib import Path

from src.utils.llamafactory import build_export_command, build_sft_train_command


def test_build_sft_train_command() -> None:
    config = Path("configs/llamafactory/sft/smoke.yaml")
    cmd = build_sft_train_command(config_path=config)
    assert cmd == ["llamafactory-cli", "train", str(config)]


def test_build_export_command() -> None:
    config = Path("configs/llamafactory/sft/qwen3_8b_lora.yaml")
    cmd = build_export_command(config_path=config, cli_bin="lf-cli")
    assert cmd == ["lf-cli", "export", str(config)]


def test_launch_script_defaults_point_to_stage2_configs() -> None:
    launch_text = Path("scripts/launch_sft.sh").read_text(encoding="utf-8")
    export_text = Path("scripts/export_model.sh").read_text(encoding="utf-8")
    assert "configs/llamafactory/sft/smoke.yaml" in launch_text
    assert "train" in launch_text
    assert "configs/llamafactory/sft/qwen3_8b_lora.yaml" in export_text
    assert "export" in export_text
