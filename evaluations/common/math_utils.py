from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def population_std(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    center = mean(data)
    return math.sqrt(sum((value - center) ** 2 for value in data) / len(data))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def normalized_entropy(items: Iterable[str]) -> float:
    counts = Counter(items)
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def levenshtein_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def lcs_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    rows = [0] * (len(right) + 1)
    for left_item in left:
        previous = 0
        for j, right_item in enumerate(right, 1):
            saved = rows[j]
            rows[j] = previous + 1 if left_item == right_item else max(rows[j], rows[j - 1])
            previous = saved
    return 2.0 * rows[-1] / max(len(left) + len(right), 1)
