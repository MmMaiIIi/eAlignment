from pathlib import Path

from src.utils.llamafactory import build_dpo_train_command


def test_build_dpo_train_command() -> None:
    config = Path("configs/llamafactory/dpo/smoke.yaml")
    cmd = build_dpo_train_command(config_path=config)
    assert cmd == ["llamafactory-cli", "train", str(config)]


def test_launch_dpo_script_defaults_point_to_stage4_configs() -> None:
    launch_text = Path("scripts/launch_dpo.sh").read_text(encoding="utf-8")
    assert "configs/llamafactory/dpo/smoke.yaml" in launch_text
    assert "train" in launch_text
