from oracle.data import Show
from oracle.evaluate import backtest
from oracle.model import SetlistModel


def make_shows():
    return [Show(f"2026-01-{i + 1:02d}", "X", ("Open", "Hit", "Rare" if i == 0 else "Other", "Close")) for i in range(6)]


def test_staples_rank_above_rare_songs():
    m = SetlistModel().fit(make_shows())
    assert m.stats["Hit"].play_rate == 1.0
    assert m.stats["Rare"].play_rate < 0.2


def test_prediction_is_ordered_by_set_position():
    m = SetlistModel().fit(make_shows())
    pred = m.predict(length=4)
    assert pred[0] == "Open" and pred[-1] == "Close"


def test_surprise_candidates_exclude_staples():
    m = SetlistModel().fit(make_shows())
    assert "Hit" not in m.surprise_candidates()


def test_backtest_scores_between_zero_and_one():
    r = backtest(make_shows() + make_shows(), min_history=3)
    assert 0 <= r["mean_overlap"] <= 1
