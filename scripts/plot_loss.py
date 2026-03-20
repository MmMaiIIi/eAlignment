from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_trainer_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("log_history", [])
    if not isinstance(history, list):
        return []
    return [row for row in history if isinstance(row, dict)]


def _extract_points(rows: list[dict[str, Any]]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    train_points: list[tuple[float, float]] = []
    eval_points: list[tuple[float, float]] = []

    for idx, row in enumerate(rows):
        step = _safe_float(row.get("step"))
        if step is None:
            step = _safe_float(row.get("global_step"))
        if step is None:
            step = _safe_float(row.get("epoch"))
        if step is None:
            step = float(idx)

        loss = _safe_float(row.get("loss"))
        eval_loss = _safe_float(row.get("eval_loss"))
        if loss is not None:
            train_points.append((step, loss))
        if eval_loss is not None:
            eval_points.append((step, eval_loss))

    train_points.sort(key=lambda item: item[0])
    eval_points.sort(key=lambda item: item[0])
    return train_points, eval_points


def _build_summary(
    run_name: str,
    train_points: list[tuple[float, float]],
    eval_points: list[tuple[float, float]],
    source_path: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_name": run_name,
        "source": source_path,
        "counts": {
            "train_points": len(train_points),
            "eval_points": len(eval_points),
        },
    }
    if train_points:
        train_values = [value for _, value in train_points]
        summary["train_loss"] = {
            "first_step": train_points[0][0],
            "last_step": train_points[-1][0],
            "first_value": train_points[0][1],
            "last_value": train_points[-1][1],
            "min_value": min(train_values),
            "max_value": max(train_values),
        }
    if eval_points:
        eval_values = [value for _, value in eval_points]
        summary["eval_loss"] = {
            "first_step": eval_points[0][0],
            "last_step": eval_points[-1][0],
            "first_value": eval_points[0][1],
            "last_value": eval_points[-1][1],
            "min_value": min(eval_values),
            "max_value": max(eval_values),
        }
    return summary


def plot_loss(
    run_name: str,
    training_log_path: Path,
    trainer_state_path: Path | None,
    output_png: Path,
    output_json: Path | None,
) -> dict[str, Any]:
    rows = _load_jsonl(training_log_path)
    source_path = str(training_log_path)
    if not rows and trainer_state_path is not None and trainer_state_path.exists():
        rows = _load_trainer_state(trainer_state_path)
        source_path = f"{trainer_state_path}#log_history"
    train_points, eval_points = _extract_points(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_points:
        ax.plot(
            [step for step, _ in train_points],
            [value for _, value in train_points],
            label="train_loss",
            linewidth=1.8,
        )
    if eval_points:
        ax.plot(
            [step for step, _ in eval_points],
            [value for _, value in eval_points],
            label="eval_loss",
            linewidth=1.8,
        )

    ax.set_title(f"SFT Loss Curve ({run_name})")
    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    if train_points or eval_points:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No loss points found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    summary = _build_summary(run_name, train_points, eval_points, source_path)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT loss curves from trainer logs.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--training-log", type=Path, required=True, help="Path to training_log.jsonl")
    parser.add_argument("--trainer-state", type=Path, default=None, help="Optional trainer_state.json path")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = plot_loss(
        run_name=args.run_name,
        training_log_path=args.training_log,
        trainer_state_path=args.trainer_state,
        output_png=args.output_png,
        output_json=args.output_json,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
