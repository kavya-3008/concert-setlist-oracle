import pytest

from oracle.data import Show
from oracle.ensemble import EnsembleModel
from oracle.recency import RecencyModel


def tour(pattern):
    return [Show(f"2026-03-{i + 1:02d}", "X", ("Core", "Alt") if p else ("Core",)) for i, p in enumerate(pattern)]


def test_single_decay_equals_recency():
    shows = tour([1, 0, 0, 1, 1, 0, 1])
    e = EnsembleModel(decays=(0.7,)).fit(shows)
    r = RecencyModel(0.7).fit(shows)
    assert e.probability("Alt") == pytest.approx(r.probability("Alt"))


def test_probability_is_mean_of_members():
    shows = tour([1, 1, 0, 0, 0, 1])
    e = EnsembleModel(decays=(0.5, 1.0)).fit(shows)
    want = (RecencyModel(0.5).fit(shows).probability("Alt") + RecencyModel(1.0).fit(shows).probability("Alt")) / 2
    assert e.probability("Alt") == pytest.approx(want)


def test_weights_shift_toward_heavier_member():
    shows = tour([1, 1, 1, 0, 0, 0])
    fast = RecencyModel(0.5).fit(shows).probability("Alt")
    slow = RecencyModel(1.0).fit(shows).probability("Alt")
    e = EnsembleModel(decays=(0.5, 1.0), weights=(9, 1)).fit(shows)
    assert fast < e.probability("Alt") < slow
    assert abs(e.probability("Alt") - fast) < abs(e.probability("Alt") - slow)


def test_predict_length_and_probability_bounds():
    shows = tour([1, 0, 1, 1, 0, 0])
    e = EnsembleModel().fit(shows)
    assert len(e.predict(length=2)) == 2
    assert all(0.0 <= e.probability(s) <= 1.0 for s in e.stats)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        EnsembleModel(decays=())
    with pytest.raises(ValueError):
        EnsembleModel(decays=(0.5, 1.0), weights=(1.0,))
