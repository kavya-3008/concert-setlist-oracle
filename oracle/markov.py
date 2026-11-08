"""Markov ordering: pick songs by recency probability, then order them by learned song-to-song transitions."""
from __future__ import annotations

from collections import defaultdict

from .data import Show
from .recency import RecencyModel


class MarkovModel(RecencyModel):
    def fit(self, shows: list[Show]) -> "MarkovModel":
        super().fit(shows)
        self.trans: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.openers: dict[str, int] = defaultdict(int)
        for show in shows:
            if show.songs:
                self.openers[show.songs[0]] += 1
            for a, b in zip(show.songs, show.songs[1:]):
                self.trans[a][b] += 1
        return self

    def predict(self, length: int | None = None) -> list[str]:
        chosen = set(super().predict(length))
        if not chosen:
            return []
        # Start from the chosen song that most often opens a show, then follow the strongest transitions.
        cur = max(chosen, key=lambda s: (self.openers.get(s, 0), -self.stats[s].mean_pos, s))
        order, remaining = [cur], chosen - {cur}
        while remaining:
            cur = max(
                remaining,
                key=lambda s: (self.trans[order[-1]].get(s, 0), -self.stats[s].mean_pos, s),
            )
            order.append(cur)
            remaining.remove(cur)
        return order
