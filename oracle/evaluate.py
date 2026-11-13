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


def order_score(predicted: list[str], actual: tuple[str, ...]) -> float:
    """Of the songs both lists share, what fraction of pairs appear in the same relative order?"""
    pos = {s: i for i, s in enumerate(actual)}
    shared = [s for s in predicted if s in pos]
    pairs = [(a, b) for i, a in enumerate(shared) for b in shared[i + 1 :]]
    if not pairs:
        return 1.0
    return sum(pos[a] < pos[b] for a, b in pairs) / len(pairs)


def compare(shows: list[Show], factories: dict, min_history: int = 5) -> dict[str, dict[str, float]]:
    """Backtest several models on identical folds. Reports song overlap and ordering quality."""
    out = {}
    for name, make in factories.items():
        overlaps, orders = [], []
        for i in range(min_history, len(shows)):
            model = make().fit(shows[:i])
            actual = shows[i].songs
            if getattr(model, "uses_city", False):
                predicted = model.predict(length=len(actual), city=shows[i].city)
            else:
                predicted = model.predict(length=len(actual))
            overlaps.append(len(set(actual) & set(predicted)) / len(actual))
            orders.append(order_score(predicted, actual))
        out[name] = {"overlap": sum(overlaps) / len(overlaps), "order": sum(orders) / len(orders)}
    return out


def slot_accuracy(shows: list[Show], make, min_history: int = 5) -> dict[str, float]:
    """How often the predicted opener and closer are exactly right, plus overall song overlap."""
    op = cl = ov = 0.0
    k = len(shows) - min_history
    if k <= 0:
        raise ValueError("not enough shows to backtest")
    for i in range(min_history, len(shows)):
        actual = shows[i].songs
        pred = make().fit(shows[:i]).predict(length=len(actual))
        op += pred[0] == actual[0]
        cl += pred[-1] == actual[-1]
        ov += len(set(actual) & set(pred)) / len(actual)
    return {"opener": op / k, "closer": cl / k, "overlap": ov / k}
