from oracle.calibration import brier, probability_pairs, reliability
from oracle.data import Show
from oracle.evaluate import compare, order_score
from oracle.markov import MarkovModel
from oracle.model import SetlistModel
from oracle.recency import RecencyModel


def drifting_tour():
    # "Old Hit" fades out, "New Hit" takes over: recency should notice, plain frequency lags.
    shows = []
    for i in range(12):
        mid = "Old Hit" if i < 6 else "New Hit"
        shows.append(Show(f"2026-01-{i + 1:02d}", "X", ("Open", mid, "Staple", "Close")))
    return shows


def test_recency_model_tracks_drift_better_than_frequency():
    shows = drifting_tour()
    assert RecencyModel(decay=0.7).fit(shows).probability("New Hit") > RecencyModel(decay=0.7).fit(shows).probability("Old Hit")
    assert SetlistModel().fit(shows).stats["New Hit"].play_rate == SetlistModel().fit(shows).stats["Old Hit"].play_rate


def test_markov_order_follows_transitions():
    shows = [Show(f"2026-02-{i + 1:02d}", "X", ("A", "B", "C", "D")) for i in range(8)]
    assert MarkovModel().fit(shows).predict(length=4) == ["A", "B", "C", "D"]


def test_order_score_perfect_and_reversed():
    assert order_score(["a", "b", "c"], ("a", "b", "c")) == 1.0
    assert order_score(["c", "b", "a"], ("a", "b", "c")) == 0.0


def test_calibration_outputs_are_sane():
    pairs = list(probability_pairs(drifting_tour(), min_history=4))
    assert 0 <= brier(pairs) <= 1
    assert all(0 <= b["observed"] <= 1 for b in reliability(pairs))


def test_compare_returns_all_models():
    res = compare(drifting_tour(), {"freq": SetlistModel, "recency": RecencyModel, "markov": MarkovModel}, min_history=4)
    assert set(res) == {"freq", "recency", "markov"}


def test_isotonic_is_monotone_and_fixes_bias():
    from oracle.calibration import IsotonicCalibrator

    # Events at p=0.2 happen half the time; at p=0.8 they happen 90% of the time.
    pairs = [(0.2, 1.0)] * 5 + [(0.2, 0.0)] * 5 + [(0.8, 1.0)] * 9 + [(0.8, 0.0)]
    cal = IsotonicCalibrator().fit(pairs)
    assert abs(cal.transform(0.2) - 0.5) < 1e-9
    assert abs(cal.transform(0.8) - 0.9) < 1e-9
    grid = [cal.transform(x / 20) for x in range(21)]
    assert grid == sorted(grid)


def test_isotonic_pools_violations():
    from oracle.calibration import IsotonicCalibrator

    cal = IsotonicCalibrator().fit([(0.1, 1.0), (0.5, 0.0), (0.9, 1.0)])
    assert cal.ys == sorted(cal.ys)


def test_recalibrated_pairs_align_and_bounded():
    from oracle.calibration import recalibrated_pairs

    raw, cal = recalibrated_pairs(drifting_tour(), min_history=3, warmup=2)
    assert len(raw) == len(cal) > 0
    assert all(0 <= p <= 1 for p, _ in cal)
    assert [y for _, y in raw] == [y for _, y in cal]


def test_slot_model_predicts_opener_and_closer_separately():
    from oracle.slots import SlotModel

    shows = [Show(f"2026-03-{i + 1:02d}", "X", ("Open", "A", "B", "Close")) for i in range(6)]
    # Most recent shows swap the closer: decay should pick it up.
    shows += [Show(f"2026-03-{i + 7:02d}", "X", ("Open", "B", "A", "New Close")) for i in range(6)]
    pred = SlotModel(decay=0.7).fit(shows).predict(length=4)
    assert pred[0] == "Open" and pred[-1] == "New Close" and len(pred) == 4 and len(set(pred)) == 4


def test_slot_accuracy_perfect_on_constant_tour():
    from oracle.evaluate import slot_accuracy
    from oracle.slots import SlotModel

    shows = [Show(f"2026-04-{i + 1:02d}", "X", ("O", "M", "C")) for i in range(8)]
    r = slot_accuracy(shows, SlotModel)
    assert r == {"opener": 1.0, "closer": 1.0, "overlap": 1.0}
