from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.eval.scoring import evaluate_response_quality


@dataclass(frozen=True)
class ComparisonThresholds:
    low_score_threshold: float = 0.6
    regression_threshold: float = 0.15


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def align_prediction_rows(
    base_rows: list[dict[str, Any]] | None, sft_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if base_rows is None:
        return [
            {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "prompt": row.get("prompt", ""),
                "base_response": None,
                "sft_response": row.get("response", ""),
            }
            for row in sft_rows
        ]

    base_by_id = {str(row.get("id", "")): row for row in base_rows}
    sft_by_id = {str(row.get("id", "")): row for row in sft_rows}
    common_ids = sorted(set(base_by_id.keys()) & set(sft_by_id.keys()))

    aligned: list[dict[str, Any]] = []
    for sample_id in common_ids:
        base = base_by_id[sample_id]
        sft = sft_by_id[sample_id]
        aligned.append(
            {
                "id": sample_id,
                "category": sft.get("category") or base.get("category", ""),
                "prompt": sft.get("prompt") or base.get("prompt", ""),
                "base_response": base.get("response", ""),
                "sft_response": sft.get("response", ""),
            }
        )
    return aligned


def evaluate_comparison_rows(
    aligned_rows: list[dict[str, Any]], thresholds: ComparisonThresholds
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    badcases: list[dict[str, Any]] = []

    for row in aligned_rows:
        base_response = row.get("base_response")
        sft_response = str(row.get("sft_response", ""))
        base_eval = evaluate_response_quality(str(base_response)) if base_response is not None else None
        sft_eval = evaluate_response_quality(sft_response)

        base_score = base_eval.proxy["proxy_score"] if base_eval is not None else None
        sft_score = sft_eval.proxy["proxy_score"]
        delta = sft_score - base_score if base_score is not None else None

        if base_score is None:
            label = "single_sft"
        elif delta > 0.03:
            label = "sft_better"
        elif delta < -0.03:
            label = "base_better"
        else:
            label = "tie"

        reasons: list[str] = []
        if sft_score < thresholds.low_score_threshold:
            reasons.append("low_proxy_score")
        if delta is not None and delta <= -thresholds.regression_threshold:
            reasons.append("regression_vs_base")
        if "policy_risk" in sft_eval.flags:
            reasons.append("policy_risk")
        if "tone_risk" in sft_eval.flags:
            reasons.append("tone_risk")
        if "generic" in sft_eval.flags or "too_short" in sft_eval.flags:
            reasons.append("low_quality")

        sample = {
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "prompt": row.get("prompt", ""),
            "base": base_eval.to_dict() if base_eval is not None else None,
            "sft": sft_eval.to_dict(),
            "delta_proxy_score": delta,
            "comparison_label": label,
            "is_badcase": len(reasons) > 0,
            "badcase_reasons": reasons,
        }
        per_sample.append(sample)
        if reasons:
            badcases.append(sample)
    return per_sample, badcases


def summarize_results(
    per_sample: list[dict[str, Any]], thresholds: ComparisonThresholds, mode: str
) -> dict[str, Any]:
    sft_scores = [float(row["sft"]["proxy_metrics"]["proxy_score"]) for row in per_sample]
    base_scores = [
        float(row["base"]["proxy_metrics"]["proxy_score"])
        for row in per_sample
        if row.get("base") is not None
    ]
    deltas = [
        float(row["delta_proxy_score"])
        for row in per_sample
        if row.get("delta_proxy_score") is not None
    ]

    wins = {"sft_better": 0, "base_better": 0, "tie": 0}
    for row in per_sample:
        label = str(row["comparison_label"])
        if label in wins:
            wins[label] += 1

    category_rows: dict[str, list[dict[str, Any]]] = {}
    for row in per_sample:
        category = str(row.get("category", "unknown") or "unknown")
        category_rows.setdefault(category, []).append(row)

    category_breakdown: dict[str, Any] = {}
    for category, rows in category_rows.items():
        c_sft = [float(r["sft"]["proxy_metrics"]["proxy_score"]) for r in rows]
        c_base = [
            float(r["base"]["proxy_metrics"]["proxy_score"])
            for r in rows
            if r.get("base") is not None
        ]
        category_breakdown[category] = {
            "samples": len(rows),
            "sft_proxy_score_avg": _mean(c_sft),
            "base_proxy_score_avg": _mean(c_base),
            "delta_proxy_score_avg": _mean(
                [float(r["delta_proxy_score"]) for r in rows if r.get("delta_proxy_score") is not None]
            ),
            "badcases": sum(bool(r.get("is_badcase")) for r in rows),
        }

    summary = {
        "mode": mode,
        "metrics_definition": {
            "exact_metrics": ["char_length", "token_count", "repetition_ratio"],
            "proxy_metrics": [
                "actionability",
                "politeness_tone",
                "policy_compliance",
                "low_quality_risk",
                "proxy_score",
            ],
            "note": "Proxy metrics are heuristic rule-based checks, not benchmark-ground-truth scores.",
        },
        "thresholds": {
            "low_score_threshold": thresholds.low_score_threshold,
            "regression_threshold": thresholds.regression_threshold,
        },
        "counts": {
            "samples": len(per_sample),
            "badcases": sum(bool(row.get("is_badcase")) for row in per_sample),
        },
        "aggregate": {
            "sft_proxy_score_avg": _mean(sft_scores),
            "base_proxy_score_avg": _mean(base_scores),
            "delta_proxy_score_avg": _mean(deltas),
            "comparison_wins": wins,
        },
        "category_breakdown": category_breakdown,
    }
    return summary


def build_markdown_report(summary: dict[str, Any], per_sample: list[dict[str, Any]]) -> str:
    top_badcases = [row for row in per_sample if bool(row.get("is_badcase"))][:5]
    lines = [
        "# Evaluation Report",
        "",
        f"- mode: {summary['mode']}",
        f"- samples: {summary['counts']['samples']}",
        f"- badcases: {summary['counts']['badcases']}",
        f"- sft_proxy_score_avg: {summary['aggregate']['sft_proxy_score_avg']:.4f}",
        f"- base_proxy_score_avg: {summary['aggregate']['base_proxy_score_avg']:.4f}",
        f"- delta_proxy_score_avg: {summary['aggregate']['delta_proxy_score_avg']:.4f}",
        "",
        "## Proxy Metrics Note",
        "These metrics are proxy heuristics and should be interpreted as directional signals.",
        "",
        "## Category Breakdown",
    ]
    for category, stats in summary["category_breakdown"].items():
        lines.append(
            f"- {category}: samples={stats['samples']}, sft_avg={stats['sft_proxy_score_avg']:.4f}, "
            f"base_avg={stats['base_proxy_score_avg']:.4f}, delta_avg={stats['delta_proxy_score_avg']:.4f}, "
            f"badcases={stats['badcases']}"
        )
    lines.extend(["", "## Top Badcases"])
    for idx, row in enumerate(top_badcases, start=1):
        lines.extend(
            [
                f"### Case {idx}",
                f"- id: {row.get('id', '')}",
                f"- category: {row.get('category', '')}",
                f"- reasons: {', '.join(row.get('badcase_reasons', []))}",
                f"- prompt: {row.get('prompt', '')}",
                f"- sft_response: {row['sft']['response']}",
                "",
            ]
        )
    if not top_badcases:
        lines.append("- No badcases under current thresholds.")
    return "\n".join(lines)
