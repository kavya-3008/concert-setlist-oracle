"""Choose the recency decay from the data, using only the shows seen so far (no leakage)."""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel

DEFAULT_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def inner_score(shows: list[Show], decay: float, min_history: int = 3) -> float:
    """Walk-forward mean overlap of RecencyModel(decay) within `shows`."""
    scores = []
    for i in range(min_history, len(shows)):
        actual = shows[i].songs
        pred = RecencyModel(decay).fit(shows[:i]).predict(length=len(actual))
        scores.append(len(set(actual) & set(pred)) / len(actual))
    return sum(scores) / len(scores)


def best_decay(shows: list[Show], grid=DEFAULT_GRID, default: float = 0.85, min_history: int = 3) -> float:
    """Decay with the best inner walk-forward score; falls back to `default` if history is too short.
    Ties go to the larger decay (the smoother, less reactive model)."""
    if len(shows) <= min_history:
        return default
    return max(grid, key=lambda d: (round(inner_score(shows, d, min_history), 12), d))


class TunedRecencyModel(RecencyModel):
    """RecencyModel whose decay is re-selected at every fit from the training shows alone."""

    def __init__(self, grid=DEFAULT_GRID):
        super().__init__()
        self.grid = grid

    def fit(self, shows: list[Show]) -> "TunedRecencyModel":
        self.decay = best_decay(shows, self.grid)
        return super().fit(shows)
