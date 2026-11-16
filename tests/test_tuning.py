from oracle.data import Show
from oracle.tuning import TunedRecencyModel, best_decay, inner_score


def drifting(n=14):
    return [Show(f"2026-01-{i + 1:02d}", "X", ("Open", "Old" if i < n // 2 else "New", "Staple", "Close")) for i in range(n)]


def stable(n=12):
    return [Show(f"2026-01-{i + 1:02d}", "X", ("A", "B", "C")) for i in range(n)]


def test_short_history_uses_default():
    assert best_decay(stable(3), default=0.42) == 0.42


def test_tie_prefers_larger_decay():
    assert best_decay(stable(), grid=(0.5, 0.9, 0.7)) == 0.9


def test_drift_prefers_fast_decay():
    shows = drifting(16)
    assert inner_score(shows, 0.5) > inner_score(shows, 1.0)
    assert best_decay(shows) < 1.0


def test_tuned_model_predicts_and_uses_only_training_shows():
    shows = drifting(16)
    m = TunedRecencyModel().fit(shows)
    assert "New" in m.predict(length=4) and "Old" not in m.predict(length=4)
    assert m.decay in m.grid
