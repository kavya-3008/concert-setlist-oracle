"""Recency-weighted play probability: recent shows count more than old ones (tours drift)."""
from __future__ import annotations

from .data import Show
from .model import SetlistModel


class RecencyModel(SetlistModel):
    """Same interface as the baseline, but play probability uses exponentially decayed weights."""

    def __init__(self, decay: float = 0.85):
        super().__init__()
        self.decay = decay
        self.prob: dict[str, float] = {}

    def fit(self, shows: list[Show]) -> "RecencyModel":
        super().fit(shows)
        n = len(shows)
        weights = [self.decay ** (n - 1 - i) for i in range(n)]
        total = sum(weights)
        self.prob = {
            song: sum(w for w, show in zip(weights, shows) if song in show.songs) / total for song in self.stats
        }
        return self

    def probability(self, song: str) -> float:
        return self.prob.get(song, 0.0)

    def predict(self, length: int | None = None) -> list[str]:
        length = length or round(self.avg_len)
        ranked = sorted(self.prob, key=lambda s: (-self.prob[s], s))[:length]
        return sorted(ranked, key=lambda s: (self.stats[s].mean_pos, s))
