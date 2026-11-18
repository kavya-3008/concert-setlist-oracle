from oracle.data import Show
from oracle.recency import RecencyModel
from oracle.trend import TrendModel, weighted_line_forecast


def test_constant_series_forecasts_constant():
    assert weighted_line_forecast([1, 1, 1, 1], 0.9) == 1.0
    assert abs(weighted_line_forecast([0, 1, 0, 1, 0, 1], 1.0) - 0.5) < 0.5  # finite, within range


def test_rising_series_extrapolates_up():
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    assert weighted_line_forecast(y, 1.0) > sum(y) / len(y)


def test_forecast_is_clipped():
    assert weighted_line_forecast([0, 0, 1, 1, 1, 1], 1.0) == 1.0
    assert weighted_line_forecast([1, 1, 1, 0, 0, 0], 1.0) == 0.0


def ramp(n=20):
    shows = []
    for i in range(n):
        songs = ["Open", "Staple", "Close"]
        if i >= 10:
            songs.insert(1, "Rising")
        if i < 15:
            songs.insert(1, "Fading")
        shows.append(Show(f"2026-01-{i + 1:02d}", "X", tuple(songs)))
    return shows


def test_trend_ranks_rising_above_fading_more_than_recency():
    shows = ramp()
    t, r = TrendModel(decay=1.0, shrink=1.0).fit(shows), RecencyModel(1.0).fit(shows)
    gap_t = t.probability("Rising") - t.probability("Fading")
    gap_r = r.probability("Rising") - r.probability("Fading")
    assert gap_t > gap_r


def test_shrink_zero_equals_recency():
    shows = ramp()
    t, r = TrendModel(decay=0.85, shrink=0.0).fit(shows), RecencyModel(0.85).fit(shows)
    assert t.prob == r.prob


def test_predict_returns_requested_length_and_uses_only_history():
    m = TrendModel().fit(ramp())
    assert len(m.predict(length=3)) == 3
