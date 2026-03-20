from pathlib import Path

from align.common import build_export_command, build_train_command, missing_required_fields
from align.config import load_profile, load_yaml, resolve


def test_profile_loading_and_defaults() -> None:
    profile_name, profile = load_profile(None)
    assert profile_name == "smoke"
    for key in ["split", "sft_config", "dpo_config", "eval_config"]:
        assert key in profile
    lora_nodeepspeed_profile_name, lora_nodeepspeed_profile = load_profile("sft_lora_nodeepspeed")
    assert lora_nodeepspeed_profile_name == "sft_lora_nodeepspeed"
    assert lora_nodeepspeed_profile["sft_config"] == "configs/sft_lora_nodeepspeed.yaml"
    plain_profile_name, plain_profile = load_profile("sft_plain")
    assert plain_profile_name == "sft_plain"
    assert plain_profile["sft_config"] == "configs/sft_plain.yaml"


def test_config_files_load_and_have_required_fields() -> None:
    sft_cfg = load_yaml("configs/sft.yaml")
    sft_lora_nodeepspeed_cfg = load_yaml("configs/sft_lora_nodeepspeed.yaml")
    sft_lora_nodeepspeed_smoke_cfg = load_yaml("configs/sft_lora_nodeepspeed_smoke.yaml")
    sft_plain_cfg = load_yaml("configs/sft_plain.yaml")
    sft_plain_smoke_cfg = load_yaml("configs/sft_plain_smoke.yaml")
    dpo_cfg = load_yaml("configs/dpo.yaml")
    assert not missing_required_fields(sft_cfg, stage="sft")
    assert not missing_required_fields(sft_lora_nodeepspeed_cfg, stage="sft")
    assert not missing_required_fields(sft_lora_nodeepspeed_smoke_cfg, stage="sft")
    assert not missing_required_fields(sft_plain_cfg, stage="sft")
    assert not missing_required_fields(sft_plain_smoke_cfg, stage="sft")
    assert not missing_required_fields(dpo_cfg, stage="dpo")
    assert load_yaml("configs/eval.yaml")
    assert load_yaml("configs/profiles.yaml")
    assert load_yaml("configs/ablations.yaml")


def test_command_builders() -> None:
    train = build_train_command("configs/sft.yaml")
    export = build_export_command("configs/sft.yaml", cli_bin="lf-cli")
    assert train == ["llamafactory-cli", "train", "configs/sft.yaml"]
    assert export == ["lf-cli", "export", "configs/sft.yaml"]


def test_core_artifacts_exist() -> None:
    for path in [
        "reports/templates/experiment_report_template.md",
        "reports/templates/badcase_analysis_template.md",
        "reports/badcases/WORKFLOW.md",
    ]:
        assert resolve(path).exists()
