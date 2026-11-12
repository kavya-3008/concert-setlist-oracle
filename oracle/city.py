"""City effects: do tours rotate songs by region? Shrinks per-city play rates toward the global recency rate."""
from __future__ import annotations

from collections import defaultdict

from .data import Show
from .recency import RecencyModel


class CityModel(RecencyModel):
    """Recency model whose probabilities are blended with the target city's own history.

    p = (n_city * p_city + k * p_global) / (n_city + k), where n_city is the number of past shows
    in that city and k is the shrinkage strength (higher = trust the global rate more).
    With no history for a city it falls back to the plain recency model.
    """

    uses_city = True

    def __init__(self, decay: float = 0.85, k: float = 3.0):
        super().__init__(decay)
        self.k = k
        self._by_city: dict[str, list[Show]] = defaultdict(list)

    def fit(self, shows: list[Show]) -> "CityModel":
        super().fit(shows)
        self._by_city = defaultdict(list)
        for s in shows:
            self._by_city[s.city].append(s)
        return self

    def city_probability(self, song: str, city: str | None) -> float:
        base = self.probability(song)
        history = self._by_city.get(city or "", [])
        if not history:
            return base
        n = len(history)
        p_city = sum(song in s.songs for s in history) / n
        return (n * p_city + self.k * base) / (n + self.k)

    def predict(self, length: int | None = None, city: str | None = None) -> list[str]:
        length = length or round(self.avg_len)
        ranked = sorted(self.prob, key=lambda s: (-self.city_probability(s, city), s))[:length]
        return sorted(ranked, key=lambda s: (self.stats[s].mean_pos, s))


def city_lift(shows: list[Show], min_shows: int = 2) -> list[tuple[str, str, float, float]]:
    """(city, song, city_rate, overall_rate) sorted by largest absolute gap; shows regional rotation."""
    overall = {}
    for s in shows:
        for song in set(s.songs):
            overall[song] = overall.get(song, 0) + 1
    by_city: dict[str, list[Show]] = defaultdict(list)
    for s in shows:
        by_city[s.city].append(s)
    out = []
    for city, ss in by_city.items():
        if len(ss) < min_shows:
            continue
        for song, tot in overall.items():
            cr = sum(song in s.songs for s in ss) / len(ss)
            out.append((city, song, cr, tot / len(shows)))
    return sorted(out, key=lambda r: -abs(r[2] - r[3]))
