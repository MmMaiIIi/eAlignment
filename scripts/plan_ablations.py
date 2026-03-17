from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.llamafactory import build_dpo_train_command, build_sft_train_command  # noqa: E402
from src.utils.paths import from_root  # noqa: E402


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else from_root(path_str)


def build_plan(config: dict[str, Any], check_paths: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for exp in config.get("experiments", []):
        train = exp.get("train", {})
        stage = str(train.get("stage", "")).lower()
        train_config = str(train.get("config", ""))

        if stage == "sft":
            train_cmd = build_sft_train_command(train_config)
        elif stage == "dpo":
            train_cmd = build_dpo_train_command(train_config)
        else:
            train_cmd = ["<unsupported-stage>", stage, train_config]

        eval_cfg = exp.get("eval", {})
        eval_cmd = [
            "python",
            "scripts/run_eval.py",
            "--config",
            str(eval_cfg.get("config", "configs/eval/comparison_eval.yaml")),
        ]
        if eval_cfg.get("base_predictions"):
            eval_cmd += ["--base-predictions", str(eval_cfg["base_predictions"])]
        if eval_cfg.get("sft_predictions"):
            eval_cmd += ["--sft-predictions", str(eval_cfg["sft_predictions"])]

        missing_paths: list[str] = []
        if check_paths:
            candidate_paths = [train_config]
            if eval_cfg.get("config"):
                candidate_paths.append(str(eval_cfg["config"]))
            if eval_cfg.get("base_predictions"):
                candidate_paths.append(str(eval_cfg["base_predictions"]))
            if eval_cfg.get("sft_predictions"):
                candidate_paths.append(str(eval_cfg["sft_predictions"]))
            for raw_path in candidate_paths:
                resolved = _resolve_path(raw_path)
                if not resolved.exists():
                    missing_paths.append(raw_path)

        rows.append(
            {
                "id": exp.get("id", ""),
                "title": exp.get("title", ""),
                "objective": exp.get("objective", ""),
                "train_stage": stage,
                "train_config": train_config,
                "train_command": train_cmd,
                "eval_command": eval_cmd,
                "notes": exp.get("notes", ""),
                "missing_paths": missing_paths,
            }
        )
    return {"name": config.get("name", "ablation_plan"), "experiments": rows}


def to_markdown(plan: dict[str, Any]) -> str:
    lines = ["# Ablation Plan", "", f"- name: {plan['name']}", ""]
    for idx, exp in enumerate(plan["experiments"], start=1):
        lines.extend(
            [
                f"## Experiment {idx}: {exp['title']}",
                f"- id: {exp['id']}",
                f"- objective: {exp['objective']}",
                f"- train_stage: {exp['train_stage']}",
                f"- train_config: {exp['train_config']}",
                f"- train_command: {' '.join(exp['train_command'])}",
                f"- eval_command: {' '.join(exp['eval_command'])}",
                f"- notes: {exp['notes']}",
                f"- missing_paths: {', '.join(exp['missing_paths']) if exp['missing_paths'] else 'none'}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight ablation command plan.")
    parser.add_argument(
        "--config", type=Path, default=from_root("configs", "eval", "ablation_matrix.yaml")
    )
    parser.add_argument(
        "--output-md", type=Path, default=from_root("reports", "experiments", "ablation_plan.md")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=from_root("reports", "experiments", "ablation_plan.json"),
    )
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    plan = build_plan(config, check_paths=args.check_paths)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(plan), encoding="utf-8")
    args.output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote ablation markdown -> {args.output_md}")
    print(f"Wrote ablation json -> {args.output_json}")


if __name__ == "__main__":
    main()
