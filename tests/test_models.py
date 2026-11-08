from oracle.calibration import brier, probability_pairs, reliability
from oracle.data import Show
from oracle.evaluate import compare, order_score
from oracle.markov import MarkovModel
from oracle.model import SetlistModel
from oracle.recency import RecencyModel


def drifting_tour():
    # "Old Hit" fades out, "New Hit" takes over: recency should notice, plain frequency lags.
    shows = []
    for i in range(12):
        mid = "Old Hit" if i < 6 else "New Hit"
        shows.append(Show(f"2026-01-{i + 1:02d}", "X", ("Open", mid, "Staple", "Close")))
    return shows


def test_recency_model_tracks_drift_better_than_frequency():
    shows = drifting_tour()
    assert RecencyModel(decay=0.7).fit(shows).probability("New Hit") > RecencyModel(decay=0.7).fit(shows).probability("Old Hit")
    assert SetlistModel().fit(shows).stats["New Hit"].play_rate == SetlistModel().fit(shows).stats["Old Hit"].play_rate


def test_markov_order_follows_transitions():
    shows = [Show(f"2026-02-{i + 1:02d}", "X", ("A", "B", "C", "D")) for i in range(8)]
    assert MarkovModel().fit(shows).predict(length=4) == ["A", "B", "C", "D"]


def test_order_score_perfect_and_reversed():
    assert order_score(["a", "b", "c"], ("a", "b", "c")) == 1.0
    assert order_score(["c", "b", "a"], ("a", "b", "c")) == 0.0


def test_calibration_outputs_are_sane():
    pairs = list(probability_pairs(drifting_tour(), min_history=4))
    assert 0 <= brier(pairs) <= 1
    assert all(0 <= b["observed"] <= 1 for b in reliability(pairs))


def test_compare_returns_all_models():
    res = compare(drifting_tour(), {"freq": SetlistModel, "recency": RecencyModel, "markov": MarkovModel}, min_history=4)
    assert set(res) == {"freq", "recency", "markov"}
