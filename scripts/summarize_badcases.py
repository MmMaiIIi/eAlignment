from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.jsonl import read_jsonl
from src.utils.paths import from_root


def summarize(rows: List[Dict[str, object]], top_k: int) -> str:
    badcases = [row for row in rows if bool(row.get("is_badcase", False))]
    badcases.sort(key=lambda row: float(row["proxy"]["score"]))  # type: ignore[index]
    selected = badcases[:top_k]

    lines = [
        "# Badcase Summary",
        "",
        f"- total_rows: {len(rows)}",
        f"- badcases: {len(badcases)}",
        "",
    ]

    for idx, row in enumerate(selected, start=1):
        proxy = row["proxy"]  # type: ignore[assignment]
        lines.extend(
            [
                f"## Case {idx}",
                f"- id: {row.get('id', '')}",
                f"- category: {row.get('category', '')}",
                f"- score: {float(proxy['score']):.4f}",
                f"- prompt: {row.get('prompt', '')}",
                f"- response: {row.get('response', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize low-scoring proxy badcases.")
    parser.add_argument(
        "--eval-file", type=Path, default=from_root("reports", "experiments", "latest_eval.jsonl")
    )
    parser.add_argument(
        "--output-md", type=Path, default=from_root("reports", "badcases", "latest_badcases.md")
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(args.eval_file)
    markdown = summarize(rows, args.top_k)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote badcase summary -> {args.output_md}")


if __name__ == "__main__":
    main()
