"""Multi-timescale ensemble: average recency probabilities across several decay rates.

A single decay forces one guess about how fast a tour drifts. Averaging fast, medium and
no-decay models hedges that guess without an inner tuning loop (compare TunedRecencyModel).
"""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel


class EnsembleModel(RecencyModel):
    """Probability is the (optionally weighted) mean of RecencyModel probabilities at each decay."""

    def __init__(self, decays: tuple[float, ...] = (0.6, 0.85, 1.0), weights: tuple[float, ...] | None = None):
        if not decays:
            raise ValueError("need at least one decay")
        if weights is not None and (len(weights) != len(decays) or sum(weights) <= 0):
            raise ValueError("weights must match decays and sum to a positive number")
        super().__init__(decays[0])
        self.decays = tuple(decays)
        self.weights = tuple(weights) if weights is not None else (1.0,) * len(decays)

    def fit(self, shows: list[Show]) -> "EnsembleModel":
        members = [RecencyModel(d).fit(shows) for d in self.decays]
        total = sum(self.weights)
        super().fit(shows)  # populates stats / avg_len
        self.prob = {
            song: sum(w * m.prob[song] for w, m in zip(self.weights, members)) / total for song in self.stats
        }
        return self
