"""Paired bootstrap: is a model's gain over a reference real, or within noise?

Many README findings say "within noise on 31 shows" by eye. Here each model's per-show overlap is
paired with the reference model's on the same folds, and the shows are resampled with replacement
to put a confidence interval on the mean difference.
"""
from __future__ import annotations

import random

from .data import Show


def per_show_overlaps(shows: list[Show], make, min_history: int = 5) -> list[float]:
    """Song-overlap of the model's prediction for each held-out show (earlier shows only)."""
    out = []
    for i in range(min_history, len(shows)):
        model = make().fit(shows[:i])
        actual = shows[i].songs
        if getattr(model, "uses_city", False):
            predicted = model.predict(length=len(actual), city=shows[i].city)
        else:
            predicted = model.predict(length=len(actual))
        out.append(len(set(actual) & set(predicted)) / len(actual))
    return out


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict[str, float]:
    """Mean of (a - b) over folds with a percentile bootstrap interval, resampling folds."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty score lists")
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    tail = (1 - level) / 2
    lo = means[int(tail * n_boot)]
    hi = means[min(n_boot - 1, int((1 - tail) * n_boot))]
    return {"diff": sum(diffs) / n, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0, "n": n}


def compare_to_reference(shows: list[Show], factories: dict, reference: str, **kw) -> dict[str, dict[str, float]]:
    """Paired difference in overlap (model minus reference) for every other model."""
    scores = {name: per_show_overlaps(shows, make) for name, make in factories.items()}
    if reference not in scores:
        raise KeyError(reference)
    return {name: paired_bootstrap(s, scores[reference], **kw) for name, s in scores.items() if name != reference}
