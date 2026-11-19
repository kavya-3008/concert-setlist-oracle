import pytest

from oracle.data import Show
from oracle.explain import Evidence, GeminiExplainer, TemplateExplainer, explain_all, gather_evidence


def tour():
    rows = [
        ("Open", "Staple", "Rare"),
        ("Open", "Staple"),
        ("Open", "Staple"),
        ("Open", "Staple", "Rare"),
        ("Open", "Staple"),
        ("Open", "Staple"),
    ]
    return [Show(f"2026-01-{i + 1:02d}", "X", r) for i, r in enumerate(rows)]


def test_evidence_counts():
    ev = {e.song: e for e in gather_evidence(tour(), ["Open", "Rare", "Ghost"])}
    assert ev["Open"].played == 6 and ev["Open"].last_gap == 0
    assert ev["Rare"].played == 2 and ev["Rare"].last_gap == 2
    assert ev["Rare"].recent_played == 1 and ev["Rare"].window == 5
    assert ev["Ghost"].played == 0 and ev["Ghost"].last_gap is None and ev["Ghost"].probability == 0.0


def test_template_mentions_real_numbers():
    ev = gather_evidence(tour(), ["Rare"])[0]
    text = TemplateExplainer().explain(ev)
    assert "2 of 6" in text and "2 shows ago" in text and "rare song" in text


def test_template_handles_unseen_and_staple():
    t = TemplateExplainer()
    assert "never played" in t.explain(gather_evidence(tour(), ["Ghost"])[0])
    assert "staple" in t.explain(gather_evidence(tour(), ["Open"])[0])


def test_gemini_uses_injected_generate_and_sees_facts():
    seen = []

    def fake(prompt):
        seen.append(prompt)
        return "  Because it always plays.  "

    ev = gather_evidence(tour(), ["Open"])[0]
    assert GeminiExplainer(fake).explain(ev) == "Because it always plays."
    assert "Played 6 of 6" in seen[0]


def test_gemini_falls_back_on_error_or_empty(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ev = gather_evidence(tour(), ["Open"])[0]
    expected = TemplateExplainer().explain(ev)
    assert GeminiExplainer().explain(ev) == expected  # no key -> template
    def boom(_):
        raise TimeoutError
    assert GeminiExplainer(boom).explain(ev) == expected
    assert GeminiExplainer(lambda _: "   ").explain(ev) == expected


def test_explain_all_and_empty():
    assert len(explain_all(tour(), ["Open", "Rare"])) == 2
    with pytest.raises(ValueError):
        gather_evidence([], ["x"])
