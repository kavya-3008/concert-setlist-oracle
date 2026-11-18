"""Trend model: extrapolate each song's play probability instead of just averaging it.

Recency weighting estimates where a song's rate *was*. When songs are being phased in or out
steadily, the next show is better predicted by where the rate is *heading*. For each song this
fits a decay-weighted least-squares line to its 0/1 presence over show index and evaluates it
one step past the last show, clipped to [0, 1].
"""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel


def weighted_line_forecast(y: list[float], decay: float) -> float:
    """Forecast y at index len(y) from a decay-weighted straight-line fit (newest show weighs most)."""
    n = len(y)
    w = [decay ** (n - 1 - i) for i in range(n)]
    sw = sum(w)
    mx = sum(wi * i for i, wi in enumerate(w)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    sxx = sum(wi * (i - mx) ** 2 for i, wi in enumerate(w))
    if sxx < 1e-12:
        return my
    slope = sum(wi * (i - mx) * (yi - my) for i, (wi, yi) in enumerate(zip(w, y))) / sxx
    return min(1.0, max(0.0, my + slope * (n - mx)))


class TrendModel(RecencyModel):
    """RecencyModel whose probabilities are one-step-ahead linear-trend forecasts.

    `shrink` blends the forecast with the plain recency probability (0 = recency only, 1 = pure trend).
    """

    def __init__(self, decay: float = 0.9, shrink: float = 0.5):
        super().__init__(decay)
        self.shrink = shrink

    def fit(self, shows: list[Show]) -> "TrendModel":
        super().fit(shows)  # sets stats, avg_len and recency probabilities
        for song in list(self.prob):
            y = [1.0 if song in show.songs else 0.0 for show in shows]
            trend = weighted_line_forecast(y, self.decay)
            self.prob[song] = (1 - self.shrink) * self.prob[song] + self.shrink * trend
        return self
