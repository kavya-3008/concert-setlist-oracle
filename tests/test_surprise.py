import pytest

from oracle.data import Show
from oracle.surprise import RANKERS, gap_ranking, recency_ranking, surprise_backtest


def make(n=12):
    # "Staple" always plays; "Rare" plays every 4th show; "Never" appears once early.
    shows = []
    for i in range(n):
        songs = ["Staple"]
        if i % 4 == 0:
            songs.append("Rare")
        if i == 1:
            songs.append("Never")
        shows.append(Show(f"2026-01-{i + 1:02d}", "X", tuple(songs)))
    return shows


def test_staples_excluded_from_pool():
    assert "Staple" not in gap_ranking(make(), 0.05, 0.5)
    assert "Staple" not in recency_ranking(make(), 0.05, 0.5)


def test_gap_ranking_puts_longest_absent_first():
    shows = make(11)  # Rare last played at index 8, Never at 1
    assert gap_ranking(shows, 0.05, 0.5)[0] == "Never"


def test_precision_and_base_rate_are_probabilities():
    for ranker in RANKERS.values():
        r = surprise_backtest(make(), ranker, k=1, min_history=4)
        assert 0 <= r["precision"] <= 1 and 0 <= r["base_rate"] <= 1
        assert r["shows_scored"] == 8


def test_periodic_song_is_found_by_gap():
    # A (period 3) and B (period 4) rotate; the longest-gap pick is due more often than a random pool pick.
    shows = [Show(f"2026-02-{i + 1:02d}", "X", ("Staple",) + (("A",) if i % 3 == 0 else ()) + (("B",) if i % 4 == 1 else ())) for i in range(36)]
    r = surprise_backtest(shows, gap_ranking, k=1, min_history=8)
    assert r["precision"] > r["base_rate"]


def test_too_few_shows_raises():
    with pytest.raises(ValueError):
        surprise_backtest(make(3), gap_ranking, min_history=5)
