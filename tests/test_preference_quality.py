from src.data.preference_quality import analyze_preference_quality


def test_preference_quality_detects_duplicates_and_imbalance() -> None:
    valid = [
        {
            "id": "pref_001",
            "category": "returns_refunds",
            "prompt": "Customer asks for refund",
            "chosen": "Please provide your order number and photo.",
            "rejected": "No.",
        },
        {
            "id": "pref_002",
            "category": "returns_refunds",
            "prompt": "Customer asks for refund",
            "chosen": "Please provide your order number and photo.",
            "rejected": "No.",
        },
        {
            "id": "pref_003",
            "category": "shipping_logistics",
            "prompt": "Where is my order?",
            "chosen": "Please share your order number and I will check.",
            "rejected": "Wait.",
        },
    ]
    rejected = [
        {"errors": ["chosen: must be non-empty"]},
        {"errors": ["chosen/rejected: chosen and rejected must differ"]},
    ]
    report = analyze_preference_quality(raw_count=5, valid_records=valid, rejected_records=rejected)
    assert report["issue_counts"]["empty_chosen_or_rejected"] == 1
    assert report["issue_counts"]["chosen_identical_to_rejected"] == 1
    assert report["duplicate_patterns"]["duplicate_prompt_groups"] >= 1
    assert report["duplicate_patterns"]["duplicate_pair_groups"] >= 1
    assert report["category_imbalance"]["max_count"] >= report["category_imbalance"]["min_count"]
