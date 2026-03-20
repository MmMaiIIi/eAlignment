from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.data import CATEGORY_INFERENCE_RULES, FALLBACK_CATEGORY
from align.io import read_jsonl, write_jsonl

PLACEHOLDER_PATTERNS = (
    re.compile(r"#\s*e-[^\s,，。!?？!;；]*", flags=re.IGNORECASE),
    re.compile(r"#\s*e-s\[[^\]]*\]", flags=re.IGNORECASE),
    re.compile(r"\[[^\]]*[xX][^\]]*\]"),
    re.compile(r"\[(?:数字|姓名|Name|name|ORDERID)[^\]]*\]"),
)

GENERIC_CLOSING_PHRASES = {
    "请问还有其他可以帮到您的吗",
    "请问还有其他还可以帮到您的吗",
    "请问还有什么可以帮您",
    "还有其他问题吗",
    "还有什么问题吗",
    "感谢您的咨询",
    "感谢您对京东的支持，祝您生活愉快，再见",
    "祝您生活愉快",
    "如有问题随时联系",
    "谢谢您的支持",
    "后期有问题再来咨询妹子哦",
    "没有其他问题妹子就不打扰您了哦",
    "没有其他问题的话妹子就和您说再见了哦",
}

SUSPICIOUS_PATTERNS = ["[数字x]", "[姓名x]", "[Name]", "#E-s", "<think>"]
BUSINESS_SIGNAL_KEYWORDS = [
    "订单",
    "退款",
    "退货",
    "物流",
    "发货",
    "配送",
    "地址",
    "售后",
    "维修",
    "投诉",
    "商品",
    "规格",
    "库存",
    "申请",
    "核实",
    "提供",
    "处理",
    "照片",
    "工单",
    "进度",
    "补发",
    "order",
    "refund",
    "return",
    "shipping",
    "delivery",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _compact(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.lower(), flags=re.UNICODE)


GENERIC_CLOSING_SIGNATURES = {_compact(text) for text in GENERIC_CLOSING_PHRASES}
GENERIC_CLOSING_REGEXES = (
    re.compile(r"^请问还有.*帮.*您.*吗[?？]*$"),
    re.compile(r"^还有.*问题.*吗[?？]*$"),
    re.compile(r"^感谢.*支持.*(再见|愉快).*$"),
    re.compile(r"^后期有问题再来咨询.*$"),
    re.compile(r"^没有.*问题.*(不打扰|再见).*$"),
    re.compile(r"^如有问题.*联系.*$"),
)
COURTESY_TEMPLATE_REGEXES = (
    re.compile(r"^很高兴遇到您.*帮到您.*吗[?？]*$"),
    re.compile(r"^您.*客气.*$"),
    re.compile(r"^不客气.*$"),
    re.compile(r"^缘聚缘散缘如水.*$"),
    re.compile(r"^还辛苦您.*评价.*$"),
    re.compile(r"^妹子.*评价.*$"),
    re.compile(r"^遇到像您这样.*评价.*$"),
    re.compile(r"^祝您[:：]?.*(开心|愉快).*$"),
)


def _contains_placeholder(value: str) -> bool:
    return any(pattern.search(value) for pattern in PLACEHOLDER_PATTERNS)


def _is_generic_closing(value: str) -> bool:
    compact = _compact(value)
    if not compact:
        return False
    if compact in GENERIC_CLOSING_SIGNATURES:
        return True
    text = _text(value)
    if any(pattern.match(text) for pattern in GENERIC_CLOSING_REGEXES):
        return True
    if any(pattern.match(text) for pattern in COURTESY_TEMPLATE_REGEXES):
        return True
    if "评价" in text and not _has_business_signal(text):
        return True
    if not _has_business_signal(text) and any(token in text for token in ["客气", "愉快", "开心", "谢谢", "亲爱", "妹子"]):
        return True
    return False


def _has_business_signal(value: str) -> bool:
    text = _text(value).lower()
    return any(keyword in text for keyword in BUSINESS_SIGNAL_KEYWORDS)


def _infer_category(value: str) -> str:
    text = value.lower()
    for keywords, category in CATEGORY_INFERENCE_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return FALLBACK_CATEGORY


def _is_language_mixture(value: str) -> bool:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", value))
    latin_chars = len(re.findall(r"[A-Za-z]", value))
    total = max(len(value), 1)
    return chinese_chars > 0 and latin_chars > 0 and chinese_chars / total > 0.1 and latin_chars / total > 0.1


def _response_length(value: str) -> int:
    return len(_compact(value))


def _rejection_distribution(rejected_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rejected_rows:
        for err in row.get("errors", []):
            counter[str(err)] += 1
    return dict(counter.most_common())


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def _build_report(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    top_responses: list[tuple[str, int]],
    sample_rows: list[dict[str, Any]],
    bad_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# SFT Data Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total_samples: {metrics['total_samples']}")
    lines.append(f"- placeholder_leak_count: {metrics['placeholder_leak_count']}")
    lines.append(f"- short_response_count: {metrics['short_response_count']}")
    lines.append(f"- generic_closing_count: {metrics['generic_closing_count']}")
    lines.append(f"- possible_mismatch_count: {metrics['possible_mismatch_count']}")
    lines.append(f"- language_mixture_count: {metrics['language_mixture_count']}")
    lines.append(f"- empty_response_count: {metrics['empty_response_count']}")
    lines.append(f"- suspicious_pattern_hits: {metrics['suspicious_pattern_hits']}")
    lines.append("")
    lines.append("## Category Distribution")
    for category, count in metrics["category_distribution"].items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Top Repeated Responses")
    for text, count in top_responses:
        lines.append(f"- ({count}) {text}")
    lines.append("")
    lines.append("## Rejection Reason Distribution")
    if metrics["rejection_reason_distribution"]:
        for reason, count in metrics["rejection_reason_distribution"].items():
            lines.append(f"- ({count}) {reason}")
    else:
        lines.append("- unavailable")
    lines.append("")
    lines.append("## Random Sample (100)")
    lines.append("| id | category | instruction | output |")
    lines.append("|---|---|---|---|")
    for row in sample_rows:
        lines.append(
            f"| {_markdown_escape(str(row.get('id', '')))} | {_markdown_escape(str(row.get('category', '')))} | "
            f"{_markdown_escape(_text(row.get('instruction', row.get('query', ''))))} | "
            f"{_markdown_escape(_text(row.get('output', row.get('response', ''))))} |"
        )
    lines.append("")
    lines.append("## Suspicious Cases")
    lines.append(f"- bad_case_count: {len(bad_rows)}")
    lines.append("- patterns_checked: " + ", ".join(SUSPICIOUS_PATTERNS))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed SFT data quality and generate report artifacts.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--rejected", type=Path, default=Path("data/interim/sft_rejected.jsonl"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/sft_data_audit.md"))
    parser.add_argument("--badcases", type=Path, default=Path("reports/sft_bad_cases.jsonl"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--short-threshold", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    rejected_rows = read_jsonl(args.rejected) if args.rejected.exists() else []
    category_distribution = Counter(str(row.get("category", "unknown")) for row in rows)

    placeholder_leak_count = 0
    short_response_count = 0
    generic_closing_count = 0
    possible_mismatch_count = 0
    language_mixture_count = 0
    empty_response_count = 0
    suspicious_pattern_hits: Counter[str] = Counter()
    responses: Counter[str] = Counter()
    bad_rows: list[dict[str, Any]] = []

    for row in rows:
        instruction = _text(row.get("instruction", row.get("query", "")))
        input_text = _text(row.get("input", ""))
        output = _text(row.get("output", row.get("response", "")))
        category = _text(row.get("category", ""))
        full_text = f"{instruction}\n{input_text}\n{output}"

        responses[output] += 1
        flags: list[str] = []

        if not output:
            empty_response_count += 1
            flags.append("empty_response")
        if _contains_placeholder(full_text):
            placeholder_leak_count += 1
            flags.append("placeholder_leak")
        if _response_length(output) < args.short_threshold:
            short_response_count += 1
            flags.append("short_response")
        if _is_generic_closing(output):
            generic_closing_count += 1
            flags.append("generic_closing")
        inferred = _infer_category(full_text)
        if inferred and category and inferred != category:
            possible_mismatch_count += 1
            flags.append("possible_category_mismatch")
        if _is_language_mixture(full_text):
            language_mixture_count += 1
            flags.append("language_mixture")

        for token in SUSPICIOUS_PATTERNS:
            if token.lower() in full_text.lower():
                suspicious_pattern_hits[token] += 1

        if flags:
            bad_rows.append(
                {
                    "id": row.get("id"),
                    "category": category,
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "flags": flags,
                    "inferred_category": inferred,
                }
            )

    randomizer = random.Random(args.seed)
    sample_size = min(args.sample_size, len(rows))
    sample_rows = randomizer.sample(rows, sample_size) if sample_size > 0 else []
    top_responses = [(text, count) for text, count in responses.most_common(20) if text]

    metrics = {
        "total_samples": len(rows),
        "placeholder_leak_count": placeholder_leak_count,
        "short_response_count": short_response_count,
        "generic_closing_count": generic_closing_count,
        "possible_mismatch_count": possible_mismatch_count,
        "language_mixture_count": language_mixture_count,
        "empty_response_count": empty_response_count,
        "top_repeated_responses": top_responses,
        "category_distribution": dict(category_distribution.most_common()),
        "rejection_reason_distribution": _rejection_distribution(rejected_rows),
        "suspicious_pattern_hits": dict(suspicious_pattern_hits.most_common()),
    }

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(_build_report(rows, metrics, top_responses, sample_rows, bad_rows), encoding="utf-8")
    write_jsonl(args.badcases, bad_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "report_md": str(args.report_md),
                "badcases": str(args.badcases),
                "summary": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
