"""Per-slot model: opener and closer are predicted on their own, the middle by recency-weighted play rate."""
from __future__ import annotations

from collections import defaultdict

from .data import Show
from .recency import RecencyModel


def _argmax(scores: dict[str, float], exclude: set[str]) -> str | None:
    cands = [(-v, k) for k, v in scores.items() if k not in exclude]
    return min(cands)[1] if cands else None


class SlotModel(RecencyModel):
    """Opener/closer chosen by decayed slot frequency; remaining slots filled by recency probability."""

    def fit(self, shows: list[Show]) -> "SlotModel":
        super().fit(shows)
        n = len(shows)
        self.opener_p: dict[str, float] = defaultdict(float)
        self.closer_p: dict[str, float] = defaultdict(float)
        for i, show in enumerate(shows):
            w = self.decay ** (n - 1 - i)
            self.opener_p[show.songs[0]] += w
            self.closer_p[show.songs[-1]] += w
        return self

    def predict(self, length: int | None = None) -> list[str]:
        length = length or round(self.avg_len)
        if length < 2:
            return super().predict(length)
        opener = _argmax(self.opener_p, set())
        closer = _argmax(self.closer_p, {opener})
        fixed = {opener, closer}
        middle = sorted((s for s in self.prob if s not in fixed), key=lambda s: (-self.prob[s], s))[: length - 2]
        middle.sort(key=lambda s: (self.stats[s].mean_pos, s))
        return [opener, *middle, closer]
