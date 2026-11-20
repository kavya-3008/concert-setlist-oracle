"""Set-length forecasting: how many songs will tonight's show have?

The other backtests hand each model the true set length, which is unavailable in real use. This
module predicts the length from past shows and scores the full pipeline (predicted length, then
predicted songs) with precision, recall and F1, since a wrong length trades one for the other.
"""
from __future__ import annotations

from .data import Show
from .recency import RecencyModel


def mean_length(shows: list[Show]) -> float:
    return sum(len(s.songs) for s in shows) / len(shows)


def decayed_length(shows: list[Show], decay: float = 0.85) -> float:
    """Exponentially weighted mean length (newest show weighs most)."""
    n = len(shows)
    w = [decay ** (n - 1 - i) for i in range(n)]
    return sum(wi * len(s.songs) for wi, s in zip(w, shows)) / sum(w)


def last_k_length(shows: list[Show], k: int = 5) -> float:
    return mean_length(shows[-k:])


def expected_size(shows: list[Show], decay: float = 0.85) -> float:
    """Sum of the recency model's per-song probabilities. This is algebraically identical to
    decayed_length (a weighted sum of indicators is a weighted set size), so it is not listed
    as a separate forecaster; a test pins the identity."""
    return sum(RecencyModel(decay).fit(shows).prob.values())


LENGTH_FORECASTERS = {
    "Overall mean": mean_length,
    "Decayed mean (0.85)": decayed_length,
    "Last 5 shows": last_k_length,
}


def f1(actual: set[str], predicted: set[str]) -> tuple[float, float, float]:
    hits = len(actual & predicted)
    p = hits / len(predicted) if predicted else 0.0
    r = hits / len(actual) if actual else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def length_backtest(shows: list[Show], forecast, min_history: int = 5, make=RecencyModel) -> dict[str, float]:
    """Walk forward: forecast the length, predict that many songs, score against the real show.

    Returns mean absolute length error, mean signed bias, and mean precision/recall/F1.
    """
    if len(shows) <= min_history:
        raise ValueError("not enough shows to backtest")
    err = bias = p = r = f = 0.0
    k = len(shows) - min_history
    for i in range(min_history, len(shows)):
        history, actual = shows[:i], shows[i].songs
        n = max(1, round(forecast(history)))
        pred = make().fit(history).predict(length=n)
        pp, rr, ff = f1(set(actual), set(pred))
        err += abs(n - len(actual))
        bias += n - len(actual)
        p, r, f = p + pp, r + rr, f + ff
    return {"mae": err / k, "bias": bias / k, "precision": p / k, "recall": r / k, "f1": f / k}
