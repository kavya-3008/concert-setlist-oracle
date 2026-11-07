"""Backtest: predict each show from the shows before it, and score the overlap."""
from __future__ import annotations

from .data import Show
from .model import SetlistModel


def backtest(shows: list[Show], min_history: int = 5) -> dict[str, float]:
    scores = []
    for i in range(min_history, len(shows)):
        model = SetlistModel().fit(shows[:i])
        actual = set(shows[i].songs)
        predicted = model.predict(length=len(shows[i].songs))
        hits = len(actual & set(predicted))
        scores.append(hits / len(actual))
    if not scores:
        raise ValueError("not enough shows to backtest")
    return {"shows_scored": len(scores), "mean_overlap": sum(scores) / len(scores)}
