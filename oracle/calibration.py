"""Are the probabilities honest? Reliability bins and Brier score from a rolling backtest."""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel


def probability_pairs(shows: list[Show], model_factory=RecencyModel, min_history: int = 5):
    """Yield (predicted_probability, did_it_happen) for every song and every held-out show."""
    for i in range(min_history, len(shows)):
        model = model_factory().fit(shows[:i])
        actual = set(shows[i].songs)
        for song in model.stats:
            yield model.probability(song), 1.0 if song in actual else 0.0


def brier(pairs) -> float:
    pairs = list(pairs)
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def reliability(pairs, bins: int = 5) -> list[dict]:
    """Group predictions into probability bins and compare predicted vs observed frequency."""
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    out = []
    for i, b in enumerate(buckets):
        if b:
            out.append(
                {
                    "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
                    "predicted": sum(p for p, _ in b) / len(b),
                    "observed": sum(y for _, y in b) / len(b),
                    "n": len(b),
                }
            )
    return out


class IsotonicCalibrator:
    """Monotone recalibration of probabilities via pool-adjacent-violators (no dependencies)."""

    def __init__(self):
        self.xs: list[float] = []
        self.ys: list[float] = []

    def fit(self, pairs) -> "IsotonicCalibrator":
        pairs = list(pairs)
        if not pairs:
            raise ValueError("need at least one (probability, outcome) pair")
        # Collapse identical probabilities first so ties are never split across blocks.
        by_p: dict[float, list[float]] = {}
        for p, y in pairs:
            agg = by_p.setdefault(p, [0.0, 0])
            agg[0] += y
            agg[1] += 1
        # Each block: [sum_y, count, sum_x]; merge neighbours while means decrease.
        blocks: list[list[float]] = []
        for p in sorted(by_p):
            sy, n = by_p[p]
            blocks.append([sy, n, p * n])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                top = blocks.pop()
                blocks[-1][0] += top[0]
                blocks[-1][1] += top[1]
                blocks[-1][2] += top[2]
        self.xs = [b[2] / b[1] for b in blocks]
        self.ys = [b[0] / b[1] for b in blocks]
        return self

    def transform(self, p: float) -> float:
        """Piecewise-linear interpolation between block centres, flat outside them."""
        xs, ys = self.xs, self.ys
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if p <= xs[i]:
                t = (p - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]


def recalibrated_pairs(shows: list[Show], model_factory=RecencyModel, min_history: int = 5, warmup: int = 3):
    """Honest evaluation: each held-out show is recalibrated using only earlier held-out folds.

    Returns (raw_pairs, calibrated_pairs) over the folds that had at least `warmup` earlier folds.
    """
    folds = []
    for i in range(min_history, len(shows)):
        model = model_factory().fit(shows[:i])
        actual = set(shows[i].songs)
        folds.append([(model.probability(s), 1.0 if s in actual else 0.0) for s in model.stats])
    raw, cal = [], []
    for k in range(warmup, len(folds)):
        cal_model = IsotonicCalibrator().fit([pr for f in folds[:k] for pr in f])
        for p, y in folds[k]:
            raw.append((p, y))
            cal.append((cal_model.transform(p), y))
    return raw, cal
