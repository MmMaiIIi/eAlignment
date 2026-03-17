from __future__ import annotations

from pathlib import Path
from typing import Any

from align.io import read_jsonl, write_json, write_jsonl


ACTION_MARKERS = ["order number", "photo", "check", "update", "track", "return", "refund", "replace"]
POLITE_MARKERS = ["sorry", "please", "understand", "thanks", "appreciate"]
AGGRESSIVE_MARKERS = ["not my fault", "your fault", "shut up", "deal with it", "cannot help"]
POLICY_RISK_MARKERS = ["off-platform payment", "private transfer", "fake tracking", "ignore policy"]
LOW_QUALITY_MARKERS = ["i do not know", "no idea", "wait", "read the page", "cannot help"]


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("\n", " ").split(" ") if token.strip()]


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def score_response(response: str) -> dict[str, Any]:
    text = response.strip()
    lowered = text.lower()
    tokens = _tokenize(text)
    repetition_ratio = _repetition_ratio(tokens)

    actionability = float(any(marker in lowered for marker in ACTION_MARKERS))
    polite = float(any(marker in lowered for marker in POLITE_MARKERS))
    tone_risk = float(any(marker in lowered for marker in AGGRESSIVE_MARKERS))
    policy_risk = float(any(marker in lowered for marker in POLICY_RISK_MARKERS))
    generic = float(any(marker in lowered for marker in LOW_QUALITY_MARKERS))
    too_short = float(len(tokens) < 6)
    low_quality = float((generic + too_short + float(repetition_ratio > 0.45)) >= 1.0)

    metrics = {
        "exact_metrics": {
            "char_length": float(len(text)),
            "token_count": float(len(tokens)),
            "repetition_ratio": repetition_ratio,
        },
        "proxy_metrics": {
            "actionability": actionability,
            "politeness_tone": float(polite == 1.0 and tone_risk == 0.0),
            "policy_compliance": float(policy_risk == 0.0 and tone_risk == 0.0),
            "low_quality_risk": low_quality,
        },
        "flags": [],
    }
    proxy = metrics["proxy_metrics"]
    proxy["proxy_score"] = (
        proxy["actionability"] + proxy["politeness_tone"] + proxy["policy_compliance"] + (1.0 - low_quality)
    ) / 4.0

    if tone_risk == 1.0:
        metrics["flags"].append("tone_risk")
    if policy_risk == 1.0:
        metrics["flags"].append("policy_risk")
    if too_short == 1.0:
        metrics["flags"].append("too_short")
    if generic == 1.0:
        metrics["flags"].append("generic")
    if repetition_ratio > 0.45:
        metrics["flags"].append("repetitive")
    return metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _align_rows(base_rows: list[dict[str, Any]] | None, tuned_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if base_rows is None:
        return [
            {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "prompt": row.get("prompt", ""),
                "base_response": None,
                "tuned_response": row.get("response", ""),
            }
            for row in tuned_rows
        ]
    base_by_id = {str(row.get("id", "")): row for row in base_rows}
    tuned_by_id = {str(row.get("id", "")): row for row in tuned_rows}
    common_ids = sorted(set(base_by_id.keys()) & set(tuned_by_id.keys()))
    return [
        {
            "id": sample_id,
            "category": tuned_by_id[sample_id].get("category") or base_by_id[sample_id].get("category", ""),
            "prompt": tuned_by_id[sample_id].get("prompt") or base_by_id[sample_id].get("prompt", ""),
            "base_response": base_by_id[sample_id].get("response", ""),
            "tuned_response": tuned_by_id[sample_id].get("response", ""),
        }
        for sample_id in common_ids
    ]


def evaluate_predictions(
    base_rows: list[dict[str, Any]] | None,
    tuned_rows: list[dict[str, Any]],
    low_score_threshold: float,
    regression_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    aligned = _align_rows(base_rows, tuned_rows)
    mode = "comparison" if base_rows is not None else "single_model"
    per_sample: list[dict[str, Any]] = []
    badcases: list[dict[str, Any]] = []

    for row in aligned:
        base_eval = score_response(str(row["base_response"])) if row["base_response"] is not None else None
        tuned_eval = score_response(str(row["tuned_response"]))
        base_score = base_eval["proxy_metrics"]["proxy_score"] if base_eval else None
        tuned_score = tuned_eval["proxy_metrics"]["proxy_score"]
        delta = tuned_score - base_score if base_score is not None else None

        if base_score is None:
            label = "single_model"
        elif delta > 0.03:
            label = "tuned_better"
        elif delta < -0.03:
            label = "base_better"
        else:
            label = "tie"

        reasons: list[str] = []
        if tuned_score < low_score_threshold:
            reasons.append("low_proxy_score")
        if delta is not None and delta <= -regression_threshold:
            reasons.append("regression_vs_base")
        if "policy_risk" in tuned_eval["flags"]:
            reasons.append("policy_risk")
        if "tone_risk" in tuned_eval["flags"]:
            reasons.append("tone_risk")
        if "generic" in tuned_eval["flags"] or "too_short" in tuned_eval["flags"]:
            reasons.append("low_quality")

        sample = {
            "id": row["id"],
            "category": row["category"],
            "prompt": row["prompt"],
            "base": {
                "response": row["base_response"],
                **base_eval,
            }
            if base_eval
            else None,
            "tuned": {"response": row["tuned_response"], **tuned_eval},
            "delta_proxy_score": delta,
            "comparison_label": label,
            "is_badcase": len(reasons) > 0,
            "badcase_reasons": reasons,
        }
        per_sample.append(sample)
        if reasons:
            badcases.append(sample)

    wins = {"tuned_better": 0, "base_better": 0, "tie": 0}
    for row in per_sample:
        tag = row["comparison_label"]
        if tag in wins:
            wins[tag] += 1

    categories: dict[str, list[dict[str, Any]]] = {}
    for row in per_sample:
        cat = str(row.get("category", "unknown") or "unknown")
        categories.setdefault(cat, []).append(row)

    category_breakdown: dict[str, Any] = {}
    for cat, rows in categories.items():
        tuned_scores = [float(item["tuned"]["proxy_metrics"]["proxy_score"]) for item in rows]
        base_scores = [
            float(item["base"]["proxy_metrics"]["proxy_score"]) for item in rows if item.get("base") is not None
        ]
        deltas = [float(item["delta_proxy_score"]) for item in rows if item["delta_proxy_score"] is not None]
        category_breakdown[cat] = {
            "samples": len(rows),
            "tuned_proxy_score_avg": _mean(tuned_scores),
            "base_proxy_score_avg": _mean(base_scores),
            "delta_proxy_score_avg": _mean(deltas),
            "badcases": sum(bool(item["is_badcase"]) for item in rows),
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
            "note": "Proxy metrics are heuristic directional checks, not benchmark-ground-truth metrics.",
        },
        "thresholds": {
            "low_score_threshold": low_score_threshold,
            "regression_threshold": regression_threshold,
        },
        "counts": {"samples": len(per_sample), "badcases": len(badcases)},
        "aggregate": {
            "tuned_proxy_score_avg": _mean(
                [float(item["tuned"]["proxy_metrics"]["proxy_score"]) for item in per_sample]
            ),
            "base_proxy_score_avg": _mean(
                [
                    float(item["base"]["proxy_metrics"]["proxy_score"])
                    for item in per_sample
                    if item.get("base") is not None
                ]
            ),
            "delta_proxy_score_avg": _mean(
                [float(item["delta_proxy_score"]) for item in per_sample if item["delta_proxy_score"] is not None]
            ),
            "comparison_wins": wins,
        },
        "category_breakdown": category_breakdown,
    }

    report = build_report_markdown(summary, per_sample)
    return summary, per_sample, badcases, report


def build_report_markdown(summary: dict[str, Any], per_sample: list[dict[str, Any]]) -> str:
    top_badcases = [row for row in per_sample if bool(row["is_badcase"])][:5]
    lines = [
        "# Evaluation Report",
        "",
        f"- mode: {summary['mode']}",
        f"- samples: {summary['counts']['samples']}",
        f"- badcases: {summary['counts']['badcases']}",
        f"- tuned_proxy_score_avg: {summary['aggregate']['tuned_proxy_score_avg']:.4f}",
        f"- base_proxy_score_avg: {summary['aggregate']['base_proxy_score_avg']:.4f}",
        f"- delta_proxy_score_avg: {summary['aggregate']['delta_proxy_score_avg']:.4f}",
        "",
        "## Proxy Metrics Note",
        "Proxy metrics are heuristic directional checks and should not be treated as benchmark truth.",
        "",
        "## Category Breakdown",
    ]
    for cat, stats in summary["category_breakdown"].items():
        lines.append(
            f"- {cat}: samples={stats['samples']}, tuned_avg={stats['tuned_proxy_score_avg']:.4f}, "
            f"base_avg={stats['base_proxy_score_avg']:.4f}, delta_avg={stats['delta_proxy_score_avg']:.4f}, "
            f"badcases={stats['badcases']}"
        )
    lines.extend(["", "## Top Badcases"])
    if not top_badcases:
        lines.append("- No badcases under current thresholds.")
    else:
        for idx, row in enumerate(top_badcases, start=1):
            lines.extend(
                [
                    f"### Case {idx}",
                    f"- id: {row.get('id', '')}",
                    f"- category: {row.get('category', '')}",
                    f"- reasons: {', '.join(row.get('badcase_reasons', []))}",
                    f"- prompt: {row.get('prompt', '')}",
                    f"- tuned_response: {row['tuned']['response']}",
                    "",
                ]
            )
    return "\n".join(lines)


def run_eval_pipeline(
    base_path: Path | None,
    tuned_path: Path,
    output_dir: Path,
    low_score_threshold: float,
    regression_threshold: float,
) -> dict[str, Any]:
    base_rows = read_jsonl(base_path) if base_path and base_path.exists() else None
    tuned_rows = read_jsonl(tuned_path)
    summary, per_sample, badcases, report = evaluate_predictions(
        base_rows=base_rows,
        tuned_rows=tuned_rows,
        low_score_threshold=low_score_threshold,
        regression_threshold=regression_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_jsonl(output_dir / "badcases.jsonl", badcases)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def summarize_badcases(rows: list[dict[str, Any]], top_k: int) -> str:
    selected = [row for row in rows if bool(row.get("is_badcase"))]
    selected.sort(key=lambda row: float(row.get("tuned", {}).get("proxy_metrics", {}).get("proxy_score", 1.0)))
    selected = selected[:top_k]

    lines = ["# Badcase Summary", "", f"- total_rows: {len(rows)}", f"- badcases: {len(selected)}", ""]
    if not selected:
        lines.append("- No badcases available.")
        return "\n".join(lines)
    for idx, row in enumerate(selected, start=1):
        tuned = row.get("tuned", {})
        lines.extend(
            [
                f"## Case {idx}",
                f"- id: {row.get('id', '')}",
                f"- category: {row.get('category', '')}",
                f"- proxy_score: {float(tuned.get('proxy_metrics', {}).get('proxy_score', 0.0)):.4f}",
                f"- reasons: {', '.join(row.get('badcase_reasons', []))}",
                f"- prompt: {row.get('prompt', '')}",
                f"- tuned_response: {tuned.get('response', '')}",
                "",
            ]
        )
    return "\n".join(lines)

