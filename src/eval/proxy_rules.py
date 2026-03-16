from typing import Dict


POLITE_MARKERS = ["sorry", "understand", "please"]
ACTION_MARKERS = ["order number", "photo", "check", "update", "next step", "help"]
BLAME_MARKERS = ["not my fault", "your fault", "cannot help", "ask someone else"]


def score_response(response: str) -> Dict[str, float]:
    text = response.lower()
    polite_tone = float(any(marker in text for marker in POLITE_MARKERS))
    actionable_next_step = float(any(marker in text for marker in ACTION_MARKERS))
    avoids_blame = float(not any(marker in text for marker in BLAME_MARKERS))
    score = (polite_tone + actionable_next_step + avoids_blame) / 3.0
    return {
        "polite_tone": polite_tone,
        "actionable_next_step": actionable_next_step,
        "avoids_blame": avoids_blame,
        "score": score,
    }
