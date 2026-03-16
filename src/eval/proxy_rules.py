from typing import Dict

from src.eval.scoring import evaluate_response_quality


def score_response(response: str) -> Dict[str, float]:
    """Backward-compatible proxy interface from Stage 0/1."""
    evaluated = evaluate_response_quality(response)
    proxy = evaluated.proxy
    return {
        "polite_tone": proxy["politeness_tone"],
        "actionable_next_step": proxy["actionability"],
        "avoids_blame": proxy["policy_compliance"],
        "score": proxy["proxy_score"],
    }
