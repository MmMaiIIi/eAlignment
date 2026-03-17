from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.config import load_profile, resolve
from align.eval import summarize_badcases
from align.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate concise badcase markdown.")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--input", type=Path, default=None, help="badcases.jsonl or per_sample.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="badcase markdown path")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    _, profile = load_profile(args.profile)
    input_path = args.input or resolve(profile.get("badcase_input", "reports/experiments/latest_eval/badcases.jsonl"))
    output_path = args.output or resolve(profile.get("badcase_output", "reports/badcases/latest_badcases.md"))

    rows = read_jsonl(input_path)
    markdown = summarize_badcases(rows, top_k=args.top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote badcase summary -> {output_path}")


if __name__ == "__main__":
    main()

