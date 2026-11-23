"""Backtest the "surprise candidates" feature, which no other module evaluates.

A surprise candidate is a rotating song (played sometimes, not a staple). The question is whether
a ranking rule picks candidates that actually get played tonight more often than a random pick
from the same rotating pool would. Each rule sees only earlier shows.
"""
from __future__ import annotations

from .data import Show
from .model import SetlistModel
from .recency import RecencyModel


def due_ranking(shows: list[Show], low: float, high: float) -> list[str]:
    """Current heuristic: rotating songs ordered by recent_rate - play_rate (most 'overdue' first)."""
    m = SetlistModel().fit(shows)
    pool = [s for s in m.stats.values() if low <= s.play_rate <= high]
    pool.sort(key=lambda s: (s.recent_rate - s.play_rate, s.song))
    return [s.song for s in pool]


def recency_ranking(shows: list[Show], low: float, high: float) -> list[str]:
    """Rotating songs ordered by recency-weighted play probability, highest first."""
    m = RecencyModel().fit(shows)
    pool = [s for s in m.stats.values() if low <= s.play_rate <= high]
    pool.sort(key=lambda s: (-m.prob[s.song], s.song))
    return [s.song for s in pool]


def gap_ranking(shows: list[Show], low: float, high: float) -> list[str]:
    """Rotating songs ordered by shows-since-last-played, longest gap first."""
    m = SetlistModel().fit(shows)
    last = {}
    for i, show in enumerate(shows):
        for s in show.songs:
            last[s] = i
    pool = [s.song for s in m.stats.values() if low <= s.play_rate <= high]
    return sorted(pool, key=lambda s: (-(len(shows) - 1 - last[s]), s))


RANKERS = {"Due heuristic (current)": due_ranking, "Recency probability": recency_ranking, "Longest gap": gap_ranking}


def surprise_backtest(shows: list[Show], ranker, k: int = 5, low: float = 0.05, high: float = 0.5, min_history: int = 5) -> dict[str, float]:
    """Precision@k of a ranker's top-k candidates versus the pool's base rate (the random-pick expectation)."""
    hits = picks = 0
    base_sum = 0.0
    folds = 0
    for i in range(min_history, len(shows)):
        ranked = ranker(shows[:i], low, high)
        if not ranked:
            continue
        played = set(shows[i].songs)
        top = ranked[:k]
        hits += sum(s in played for s in top)
        picks += len(top)
        base_sum += sum(s in played for s in ranked) / len(ranked)
        folds += 1
    if not folds:
        raise ValueError("not enough shows to backtest")
    return {"precision": hits / picks, "base_rate": base_sum / folds, "shows_scored": folds}
