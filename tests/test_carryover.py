import math

from oracle.carryover import CarryoverModel
from oracle.data import Show
from oracle.recency import RecencyModel


def tour(pattern):
    """Song 'Alt' follows `pattern` (list of 0/1); 'Core' always plays."""
    return [
        Show(f"2026-02-{i + 1:02d}", "X", ("Core", "Alt") if p else ("Core",)) for i, p in enumerate(pattern)
    ]


def test_alternating_song_is_expected_after_absence():
    shows = tour([1, 0] * 6 + [1, 0])  # last show: Alt absent, so it should be due
    m = CarryoverModel(k=0.5).fit(shows)
    assert m.probability("Alt") > 0.7
    assert RecencyModel().fit(shows).probability("Alt") < 0.6


def test_alternating_song_is_unlikely_right_after_playing():
    shows = tour([0, 1] * 6 + [0, 1])
    assert CarryoverModel(k=0.5).fit(shows).probability("Alt") < 0.3


def test_heavy_shrinkage_approaches_recency():
    shows = tour([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    m = CarryoverModel(k=1e6).fit(shows)
    r = RecencyModel().fit(shows)
    assert abs(m.probability("Alt") - r.probability("Alt")) < 1e-3


def test_carryover_raw_rates_and_nan():
    shows = tour([1, 0, 1, 0])
    hi, lo = CarryoverModel().carryover(shows, "Alt")
    assert hi == 0.0 and lo == 1.0
    hi, lo = CarryoverModel().carryover(shows, "Core")
    assert hi == 1.0 and math.isnan(lo)


def test_predict_length():
    assert len(CarryoverModel().fit(tour([1, 0, 1, 1, 0])).predict(length=2)) == 2
