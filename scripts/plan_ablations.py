from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.common import build_train_command
from align.config import load_yaml, resolve


def build_plan(config: dict[str, Any], check_paths: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for exp in config.get("experiments", []):
        stage = str(exp.get("stage", "")).lower()
        train_config = str(exp.get("train_config", ""))
        eval_config = str(exp.get("eval_config", "configs/eval.yaml"))
        base_predictions = exp.get("base_predictions")
        tuned_predictions = exp.get("tuned_predictions")

        train_cmd = build_train_command(train_config)
        eval_cmd = ["python", "scripts/eval.py", "--config", eval_config]
        if base_predictions:
            eval_cmd += ["--base", str(base_predictions)]
        if tuned_predictions:
            eval_cmd += ["--tuned", str(tuned_predictions)]

        missing_paths: list[str] = []
        if check_paths:
            candidates = [train_config, eval_config]
            if base_predictions:
                candidates.append(str(base_predictions))
            if tuned_predictions:
                candidates.append(str(tuned_predictions))
            for candidate in candidates:
                if not resolve(candidate).exists():
                    missing_paths.append(candidate)

        rows.append(
            {
                "id": exp.get("id", ""),
                "title": exp.get("title", ""),
                "objective": exp.get("objective", ""),
                "stage": stage,
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
                f"- stage: {exp['stage']}",
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
    parser = argparse.ArgumentParser(description="Generate ablation command plan.")
    parser.add_argument("--config", type=Path, default=resolve("configs/ablations.yaml"))
    parser.add_argument("--output-md", type=Path, default=resolve("reports/experiments/ablation_plan.md"))
    parser.add_argument("--output-json", type=Path, default=resolve("reports/experiments/ablation_plan.json"))
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    plan = build_plan(cfg, check_paths=args.check_paths)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(plan), encoding="utf-8")
    args.output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote ablation markdown -> {args.output_md}")
    print(f"Wrote ablation json -> {args.output_json}")


if __name__ == "__main__":
    main()

