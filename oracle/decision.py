"""Decision rules: turn per-song probabilities into a predicted set without being told the length.

Given probabilities p_i, ranking by p is right; the open question is how many songs to take.
Expected-F1 selection picks the k that maximises the standard approximation
E[F1] ~ 2 * (sum of the top-k probabilities) / (k + sum of all probabilities),
which needs no length forecast. A fixed probability threshold is the simpler alternative.
"""
from __future__ import annotations

from .data import Show
from .length import f1
from .recency import RecencyModel


def _ranked(prob: dict[str, float]) -> list[str]:
    return sorted(prob, key=lambda s: (-prob[s], s))


def expected_f1_size(prob: dict[str, float]) -> int:
    """The set size k maximising the approximate expected F1 (ties go to the smaller set)."""
    ranked = _ranked(prob)
    total = sum(prob.values())
    if not ranked or total <= 0:
        return 0
    best_k, best, running = 0, -1.0, 0.0
    for k, song in enumerate(ranked, 1):
        running += prob[song]
        score = 2 * running / (k + total)
        if score > best + 1e-12:
            best_k, best = k, score
    return best_k


def threshold_size(prob: dict[str, float], threshold: float = 0.5) -> int:
    return sum(p >= threshold for p in prob.values())


def select(model: RecencyModel, size: int) -> list[str]:
    """Top-`size` songs by probability, ordered by typical position (same as model.predict)."""
    return model.predict(length=size) if size > 0 else []


SIZE_RULES = {
    "Expected-F1 optimal": expected_f1_size,
    "Threshold 0.5": threshold_size,
    "Threshold 0.4": lambda p: threshold_size(p, 0.4),
    "Threshold 0.6": lambda p: threshold_size(p, 0.6),
}


def size_rule_backtest(shows: list[Show], rule, min_history: int = 5, decay: float = 0.85) -> dict[str, float]:
    """Walk forward with a size rule on the recency model's probabilities; no true length is used."""
    if len(shows) <= min_history:
        raise ValueError("not enough shows to backtest")
    n_folds = len(shows) - min_history
    p = r = f = bias = 0.0
    for i in range(min_history, len(shows)):
        model = RecencyModel(decay).fit(shows[:i])
        size = max(1, rule(model.prob))
        pred = select(model, size)
        pp, rr, ff = f1(set(shows[i].songs), set(pred))
        p, r, f = p + pp, r + rr, f + ff
        bias += size - len(shows[i].songs)
    return {"precision": p / n_folds, "recall": r / n_folds, "f1": f / n_folds, "bias": bias / n_folds}


def per_show_f1(shows: list[Show], size_fn, min_history: int = 5, decay: float = 0.85) -> list[float]:
    """Per-fold F1 where size_fn(history, model) -> number of songs to predict."""
    out = []
    for i in range(min_history, len(shows)):
        model = RecencyModel(decay).fit(shows[:i])
        size = max(1, size_fn(shows[:i], model))
        out.append(f1(set(shows[i].songs), set(select(model, size)))[2])
    return out
