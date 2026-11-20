import pytest

from oracle.data import Show
from oracle.length import (decayed_length, expected_size, f1, last_k_length, length_backtest,
                           mean_length)


def tour(lengths):
    songs = [f"S{i}" for i in range(20)]
    return [Show(f"2026-01-{i + 1:02d}", "X", tuple(songs[:n])) for i, n in enumerate(lengths)]


def test_mean_and_last_k():
    shows = tour([8, 8, 12, 12])
    assert mean_length(shows) == 10
    assert last_k_length(shows, 2) == 12


def test_decayed_length_leans_recent_and_decay_one_is_mean():
    shows = tour([8, 8, 8, 12])
    assert mean_length(shows) < decayed_length(shows, 0.5) < 12
    assert decayed_length(shows, 1.0) == pytest.approx(mean_length(shows))


def test_expected_size_matches_constant_length():
    assert expected_size(tour([10] * 6)) == pytest.approx(10)


def test_f1_values():
    assert f1({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)
    p, r, f = f1({"a", "b", "c", "d"}, {"a", "b"})
    assert (p, r) == (1.0, 0.5) and f == pytest.approx(2 / 3)
    assert f1({"a"}, {"b"}) == (0.0, 0.0, 0.0)


def test_backtest_perfect_on_constant_tour():
    res = length_backtest(tour([10] * 12), mean_length)
    assert res["mae"] == 0 and res["f1"] == 1.0 and res["bias"] == 0


def test_recent_forecaster_beats_mean_when_length_steps_up():
    shows = tour([8] * 10 + [13] * 10)
    assert length_backtest(shows, last_k_length)["mae"] < length_backtest(shows, mean_length)["mae"]


def test_uses_only_history():
    seen = []
    shows = tour([9, 9, 9, 9, 9, 9, 9])
    length_backtest(shows, lambda h: seen.append(len(h)) or 9)
    assert seen == [5, 6]


def test_too_few_shows():
    with pytest.raises(ValueError):
        length_backtest(tour([9] * 5), mean_length)


def test_expected_size_equals_decayed_length():
    shows = tour([8, 10, 9, 12, 11, 13, 10])
    assert expected_size(shows, 0.7) == pytest.approx(decayed_length(shows, 0.7))
