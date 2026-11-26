import pytest

from oracle.data import Show
from oracle.decision import expected_f1_size, size_rule_backtest, threshold_size


def test_certain_songs_all_taken():
    assert expected_f1_size({"a": 1.0, "b": 1.0, "c": 1.0}) == 3


def test_certain_plus_unlikely():
    assert expected_f1_size({"a": 1.0, "b": 1.0, "c": 0.05, "d": 0.05}) == 2


def test_empty_and_zero():
    assert expected_f1_size({}) == 0
    assert expected_f1_size({"a": 0.0}) == 0


def test_f1_optimal_takes_songs_below_half():
    # Many coin-flip-ish songs: F1 favours including those slightly under 0.5.
    prob = {"a": 1.0, "b": 0.45, "c": 0.45, "d": 0.45}
    assert expected_f1_size(prob) == 4


def test_threshold_size():
    prob = {"a": 0.9, "b": 0.5, "c": 0.49}
    assert threshold_size(prob) == 2
    assert threshold_size(prob, 0.95) == 0


def test_backtest_perfect_on_constant_tour():
    songs = ("A", "B", "C", "D")
    shows = [Show(f"2026-01-{i + 1:02d}", "X", songs) for i in range(10)]
    res = size_rule_backtest(shows, expected_f1_size)
    assert res["f1"] == pytest.approx(1.0) and res["bias"] == 0


def test_backtest_needs_history():
    with pytest.raises(ValueError):
        size_rule_backtest([], expected_f1_size)


def test_per_show_f1_matches_backtest_mean():
    from oracle.decision import per_show_f1
    songs = [f"S{i}" for i in range(8)]
    shows = [Show(f"2026-01-{i + 1:02d}", "X", tuple(songs[: 4 + i % 3])) for i in range(12)]
    scores = per_show_f1(shows, lambda h, m: expected_f1_size(m.prob))
    res = size_rule_backtest(shows, expected_f1_size)
    assert sum(scores) / len(scores) == pytest.approx(res["f1"])
