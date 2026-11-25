import pytest

from oracle.data import Show
from oracle.model import SetlistModel
from oracle.recency import RecencyModel
from oracle.significance import compare_to_reference, paired_bootstrap, per_show_overlaps


def test_identical_scores_give_zero_interval():
    r = paired_bootstrap([0.5, 0.7, 0.6], [0.5, 0.7, 0.6])
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 and not r["significant"]


def test_consistent_gain_is_significant():
    a = [0.8] * 20
    b = [0.7, 0.72, 0.68, 0.7] * 5
    r = paired_bootstrap(a, b)
    assert r["diff"] > 0.09 and r["significant"] and r["lo"] > 0


def test_noisy_zero_mean_difference_not_significant():
    a = [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8]
    b = [0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6]
    r = paired_bootstrap(a, b)
    assert abs(r["diff"]) < 1e-9 and not r["significant"] and r["lo"] < 0 < r["hi"]


def test_interval_contains_point_estimate_and_is_deterministic():
    a, b = [0.5, 0.9, 0.4, 0.8, 0.7], [0.6, 0.7, 0.5, 0.6, 0.7]
    r1, r2 = paired_bootstrap(a, b, seed=3), paired_bootstrap(a, b, seed=3)
    assert r1 == r2 and r1["lo"] <= r1["diff"] <= r1["hi"]


def test_bad_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.5])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def tour(n=14):
    return [Show(f"2026-01-{i + 1:02d}", "X", ("A", "B", "C") if i % 2 else ("A", "B", "D")) for i in range(n)]


def test_per_show_overlaps_length_and_range():
    s = per_show_overlaps(tour(), SetlistModel, min_history=5)
    assert len(s) == 9 and all(0 <= x <= 1 for x in s)


def test_compare_to_reference_excludes_reference():
    out = compare_to_reference(tour(), {"freq": SetlistModel, "rec": RecencyModel}, "freq", n_boot=200)
    assert list(out) == ["rec"] and out["rec"]["n"] == 9
    with pytest.raises(KeyError):
        compare_to_reference(tour(), {"freq": SetlistModel}, "nope")
