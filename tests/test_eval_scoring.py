from src.eval.scoring import evaluate_response_quality


def test_proxy_score_prefers_actionable_polite_response() -> None:
    good = "I am sorry for the delay. Please share your order number and I will check the latest update."
    weak = "Wait."
    good_eval = evaluate_response_quality(good)
    weak_eval = evaluate_response_quality(weak)
    assert good_eval.proxy["proxy_score"] > weak_eval.proxy["proxy_score"]


def test_policy_and_low_quality_flags() -> None:
    text = "Not my fault. Use off-platform payment."
    evaluated = evaluate_response_quality(text)
    assert "tone_risk" in evaluated.flags
    assert "policy_risk" in evaluated.flags
