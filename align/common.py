from __future__ import annotations

from pathlib import Path
from typing import Any


SFT_REQUIRED_FIELDS = {
    "stage",
    "do_train",
    "model_name_or_path",
    "template",
    "finetuning_type",
    "dataset",
    "dataset_dir",
    "output_dir",
}

DPO_REQUIRED_FIELDS = {
    "stage",
    "do_train",
    "model_name_or_path",
    "template",
    "finetuning_type",
    "dataset",
    "dataset_dir",
    "pref_beta",
    "output_dir",
}


def build_train_command(config_path: str | Path, cli_bin: str = "llamafactory-cli") -> list[str]:
    return [cli_bin, "train", str(config_path)]


def build_export_command(config_path: str | Path, cli_bin: str = "llamafactory-cli") -> list[str]:
    return [cli_bin, "export", str(config_path)]


def missing_required_fields(config: dict[str, Any], stage: str) -> list[str]:
    required = SFT_REQUIRED_FIELDS if stage == "sft" else DPO_REQUIRED_FIELDS
    return sorted([field for field in required if field not in config])

