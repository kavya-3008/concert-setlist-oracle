import pytest

from oracle.data import Show
from oracle.errors import error_analysis
from oracle.recency import RecencyModel


def tour():
    shows = [Show(f"2026-01-{i + 1:02d}", "X", ("A", "B", "C")) for i in range(6)]
    shows.append(Show("2026-01-07", "X", ("A", "B", "Z")))  # Z never seen: unreachable
    shows.append(Show("2026-01-08", "X", ("A", "B", "C")))
    return shows


def test_unreachable_and_ceiling():
    r = error_analysis(tour(), RecencyModel)
    assert r["shows_scored"] == 3
    assert r["unreachable_share"] == 1.0  # the only miss is the unseen song Z
    assert r["ceiling"] == pytest.approx((1 + 2 / 3 + 1) / 3)
    assert r["most_spurious"][0][0] == "C"


def test_reachable_miss_counted():
    shows = [Show(f"2026-01-{i + 1:02d}", "X", ("A", "B")) for i in range(5)]
    shows += [Show("2026-01-06", "X", ("A", "C")), Show("2026-01-07", "X", ("A", "C")), Show("2026-01-08", "X", ("A", "C"))]
    r = error_analysis(shows, lambda: RecencyModel(decay=1.0))
    assert r["most_missed"][0] == ("C", 2)  # first C is unseen, the next two are reachable misses
    assert r["unreachable_share"] < 1.0


def test_too_short():
    with pytest.raises(ValueError):
        error_analysis(tour()[:5], RecencyModel)
