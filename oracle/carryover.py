"""Carry-over model: does a song's chance depend on whether it was played at the previous show?

Bands that rotate songs on purpose avoid repeats (negative carry-over); bands that phase songs in
and out show positive carry-over (a song played last night is likely tonight). For each song this
estimates P(play | played at previous show) and P(play | absent at previous show) from the history,
shrunk toward the song's recency probability, and uses the one matching the last observed show.
"""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel


class CarryoverModel(RecencyModel):
    """`k` is the shrinkage strength (pseudo-shows) toward the recency probability."""

    def __init__(self, decay: float = 0.85, k: float = 3.0):
        super().__init__(decay)
        self.k = k

    def fit(self, shows: list[Show]) -> "CarryoverModel":
        super().fit(shows)
        last = set(shows[-1].songs)
        for song in list(self.prob):
            base = self.prob[song]
            state = song in last
            n = hits = 0
            for prev, cur in zip(shows, shows[1:]):
                if (song in prev.songs) == state:
                    n += 1
                    hits += song in cur.songs
            self.prob[song] = (hits + self.k * base) / (n + self.k)
        return self

    def carryover(self, shows: list[Show], song: str) -> tuple[float, float]:
        """Raw (P(play | played previous show), P(play | absent previous show)); NaN where undefined."""
        a = b = c = d = 0
        for prev, cur in zip(shows, shows[1:]):
            if song in prev.songs:
                a += 1
                b += song in cur.songs
            else:
                c += 1
                d += song in cur.songs
        nan = float("nan")
        return (b / a if a else nan, d / c if c else nan)
