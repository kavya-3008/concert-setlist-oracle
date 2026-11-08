"""Are the probabilities honest? Reliability bins and Brier score from a rolling backtest."""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel


def probability_pairs(shows: list[Show], model_factory=RecencyModel, min_history: int = 5):
    """Yield (predicted_probability, did_it_happen) for every song and every held-out show."""
    for i in range(min_history, len(shows)):
        model = model_factory().fit(shows[:i])
        actual = set(shows[i].songs)
        for song in model.stats:
            yield model.probability(song), 1.0 if song in actual else 0.0


def brier(pairs) -> float:
    pairs = list(pairs)
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def reliability(pairs, bins: int = 5) -> list[dict]:
    """Group predictions into probability bins and compare predicted vs observed frequency."""
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    out = []
    for i, b in enumerate(buckets):
        if b:
            out.append(
                {
                    "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
                    "predicted": sum(p for p, _ in b) / len(b),
                    "observed": sum(y for _, y in b) / len(b),
                    "n": len(b),
                }
            )
    return out
