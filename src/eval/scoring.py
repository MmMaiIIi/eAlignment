from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_MARKERS = [
    "order number",
    "photo",
    "check",
    "update",
    "track",
    "return",
    "refund",
    "replace",
    "steps",
]
POLITE_MARKERS = ["sorry", "please", "understand", "thanks", "appreciate"]
AGGRESSIVE_MARKERS = ["not my fault", "your fault", "shut up", "deal with it", "cannot help"]
POLICY_RISK_MARKERS = [
    "off-platform payment",
    "private transfer",
    "fake tracking",
    "ignore policy",
    "no return without reason",
]
GENERIC_LOW_QUALITY = [
    "i do not know",
    "no idea",
    "wait",
    "read the page",
    "cannot help",
]


@dataclass(frozen=True)
class EvaluatedResponse:
    response: str
    exact: dict[str, float]
    proxy: dict[str, float]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "exact_metrics": self.exact,
            "proxy_metrics": self.proxy,
            "flags": self.flags,
        }


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("\n", " ").split(" ") if token.strip()]


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def evaluate_response_quality(response: str) -> EvaluatedResponse:
    text = response.strip()
    lowered = text.lower()
    tokens = _tokenize(text)
    repetition_ratio = _repetition_ratio(tokens)

    exact = {
        "char_length": float(len(text)),
        "token_count": float(len(tokens)),
        "repetition_ratio": repetition_ratio,
    }

    actionability = float(any(marker in lowered for marker in ACTION_MARKERS))
    polite = float(any(marker in lowered for marker in POLITE_MARKERS))
    tone_risk = float(any(marker in lowered for marker in AGGRESSIVE_MARKERS))
    policy_risk = float(any(marker in lowered for marker in POLICY_RISK_MARKERS))
    generic_response = float(any(marker in lowered for marker in GENERIC_LOW_QUALITY))
    too_short = float(len(tokens) < 6)

    low_quality = float((generic_response + too_short + float(repetition_ratio > 0.45)) >= 1.0)
    policy_compliance = float(policy_risk == 0.0 and tone_risk == 0.0)
    politeness_tone = float(polite == 1.0 and tone_risk == 0.0)
    proxy_score = (
        actionability + politeness_tone + policy_compliance + (1.0 - low_quality)
    ) / 4.0

    proxy = {
        "actionability": actionability,
        "politeness_tone": politeness_tone,
        "policy_compliance": policy_compliance,
        "low_quality_risk": low_quality,
        "proxy_score": proxy_score,
    }

    flags: list[str] = []
    if tone_risk == 1.0:
        flags.append("tone_risk")
    if policy_risk == 1.0:
        flags.append("policy_risk")
    if too_short == 1.0:
        flags.append("too_short")
    if generic_response == 1.0:
        flags.append("generic")
    if repetition_ratio > 0.45:
        flags.append("repetitive")

    return EvaluatedResponse(response=text, exact=exact, proxy=proxy, flags=flags)
