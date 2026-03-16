from __future__ import annotations

import random
from typing import Any


def _ratio_to_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total == 0:
        return 0, 0, 0
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in range(remainder):
        counts[order[idx % 3]] += 1

    if total >= 3:
        for target in [1, 2]:
            if counts[target] == 0:
                donor = max(range(3), key=lambda idx: counts[idx])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[target] += 1
    return counts[0], counts[1], counts[2]


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    if abs((train_ratio + dev_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_train, n_dev, n_test = _ratio_to_counts(len(shuffled), (train_ratio, dev_ratio, test_ratio))

    train = shuffled[:n_train]
    dev = shuffled[n_train : n_train + n_dev]
    test = shuffled[n_train + n_dev : n_train + n_dev + n_test]
    return {"train": train, "dev": dev, "test": test}
