from __future__ import annotations

from pathlib import Path
from typing import Any


SFT_REQUIRED_FIELDS = [
    "stage",
    "do_train",
    "model_name_or_path",
    "template",
    "finetuning_type",
    "dataset",
    "dataset_dir",
    "output_dir",
]


def build_lf_command(action: str, config_path: str | Path, cli_bin: str = "llamafactory-cli") -> list[str]:
    return [cli_bin, action, str(config_path)]


def build_sft_train_command(config_path: str | Path, cli_bin: str = "llamafactory-cli") -> list[str]:
    return build_lf_command("train", config_path, cli_bin=cli_bin)


def build_export_command(config_path: str | Path, cli_bin: str = "llamafactory-cli") -> list[str]:
    return build_lf_command("export", config_path, cli_bin=cli_bin)


def missing_required_sft_fields(config: dict[str, Any]) -> list[str]:
    return [field for field in SFT_REQUIRED_FIELDS if field not in config]
