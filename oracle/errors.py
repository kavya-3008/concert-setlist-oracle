"""Error analysis: where does a model's backtest miss come from?

Misses split into two kinds. *Unreachable* misses are songs never played earlier in the tour
(no history-based model can name them). *Reachable* misses are songs the model had seen but
ranked outside the predicted set. Spurious picks are predicted songs that were not played.
"""
from __future__ import annotations

from collections import Counter

from .data import Show


def error_analysis(shows: list[Show], make, min_history: int = 5) -> dict:
    if len(shows) <= min_history:
        raise ValueError("not enough shows to backtest")
    missed: Counter = Counter()
    spurious: Counter = Counter()
    played = hits = unreachable = 0
    ceiling = 0.0
    k = len(shows) - min_history
    for i in range(min_history, len(shows)):
        seen = {s for show in shows[:i] for s in show.songs}
        actual = set(shows[i].songs)
        pred = set(make().fit(shows[:i]).predict(length=len(actual)))
        played += len(actual)
        hits += len(actual & pred)
        new = actual - seen
        unreachable += len(new)
        ceiling += (len(actual) - len(new)) / len(actual)
        for s in actual - pred - new:
            missed[s] += 1
        for s in pred - actual:
            spurious[s] += 1
    total_miss = played - hits
    return {
        "shows_scored": k,
        "overlap": hits / played,
        "ceiling": ceiling / k,  # best possible mean overlap given unseen songs
        "unreachable_share": unreachable / total_miss if total_miss else 0.0,
        "most_missed": missed.most_common(5),
        "most_spurious": spurious.most_common(5),
    }
