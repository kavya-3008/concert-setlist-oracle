from oracle.city import CityModel, city_lift
from oracle.data import Show
from oracle.evaluate import compare
from oracle.recency import RecencyModel


def regional_tour(n=24):
    shows = []
    for i in range(n):
        city = ["East", "West"][i % 2]
        songs = ["Opener", "Hit A", "Hit B", "East Special" if city == "East" else "West Special", "Closer"]
        shows.append(Show(f"2026-01-{i + 1:02d}", city, tuple(songs)))
    return shows


def test_city_model_picks_regional_song():
    shows = regional_tour()
    m = CityModel().fit(shows)
    assert "East Special" in m.predict(length=5, city="East")
    assert "West Special" in m.predict(length=5, city="West")
    assert "West Special" not in m.predict(length=5, city="East")


def test_unknown_city_falls_back_to_recency():
    shows = regional_tour()
    assert CityModel().fit(shows).predict(length=5, city="Mars") == RecencyModel().fit(shows).predict(length=5)


def test_shrinkage_toward_global_with_large_k():
    shows = regional_tour()
    m = CityModel(k=1e9).fit(shows)
    assert abs(m.city_probability("East Special", "East") - m.probability("East Special")) < 1e-6


def test_compare_uses_city_and_wins_on_regional_data():
    res = compare(regional_tour(), {"recency": RecencyModel, "city": CityModel})
    assert res["city"]["overlap"] > res["recency"]["overlap"]


def test_city_lift_ranks_regional_songs_first():
    top = city_lift(regional_tour())[0]
    assert top[1] in {"East Special", "West Special"}
    assert abs(top[2] - top[3]) == 0.5
