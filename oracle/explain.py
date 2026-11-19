"""Explanations for each pick ("why this song?").

Evidence is computed from the shows (never by an LLM), so numbers in an explanation are always real.
`TemplateExplainer` turns evidence into a sentence offline. `GeminiExplainer` asks an LLM to
rephrase the same evidence; it takes an injectable `generate` callable so it can be tested
without a key or network, and falls back to the template if the call fails.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol

from .data import Show
from .recency import RecencyModel


@dataclass(frozen=True)
class Evidence:
    song: str
    probability: float  # recency-weighted chance of being played
    play_rate: float  # plain share of shows
    played: int  # number of shows it appeared in
    shows: int  # number of shows considered
    last_gap: int | None  # shows since last played (0 = played in the latest show); None if never
    recent_played: int  # appearances in the last `window` shows
    window: int


def gather_evidence(shows: list[Show], songs: list[str], window: int = 5, decay: float = 0.85) -> list[Evidence]:
    if not shows:
        raise ValueError("need at least one show")
    model = RecencyModel(decay).fit(shows)
    recent = shows[-window:]
    out = []
    for song in songs:
        idx = [i for i, s in enumerate(shows) if song in s.songs]
        out.append(
            Evidence(
                song=song,
                probability=model.probability(song),
                play_rate=len(idx) / len(shows),
                played=len(idx),
                shows=len(shows),
                last_gap=(len(shows) - 1 - idx[-1]) if idx else None,
                recent_played=sum(song in s.songs for s in recent),
                window=len(recent),
            )
        )
    return out


class Explainer(Protocol):
    def explain(self, ev: Evidence) -> str: ...


class TemplateExplainer:
    def explain(self, ev: Evidence) -> str:
        if ev.played == 0:
            return f"{ev.song}: never played on this tour, so there is no history to predict it from."
        if ev.play_rate >= 0.9:
            kind = "a staple"
        elif ev.play_rate >= 0.4:
            kind = "a regular rotation song"
        else:
            kind = "a rare song"
        gap = (
            "played in the most recent show"
            if ev.last_gap == 0
            else f"last played {ev.last_gap} shows ago"
        )
        return (
            f"{ev.song}: {kind}. Played {ev.played} of {ev.shows} shows ({ev.play_rate:.0%}), "
            f"{ev.recent_played} of the last {ev.window}; {gap}. "
            f"Recency-weighted chance tonight: {ev.probability:.0%}."
        )


class GeminiExplainer:
    """Rephrases evidence with an LLM. `generate(prompt) -> str` is injectable; the default calls Gemini."""

    def __init__(self, generate: Callable[[str], str] | None = None, fallback: Explainer | None = None):
        self._generate = generate
        self.fallback = fallback or TemplateExplainer()

    def _default_generate(self, prompt: str) -> str:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        import google.generativeai as genai  # optional dependency, imported lazily

        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text

    def explain(self, ev: Evidence) -> str:
        facts = self.fallback.explain(ev)
        prompt = (
            "In one friendly sentence, explain why this song is predicted for tonight's show. "
            "Use only these facts and do not add any numbers:\n" + facts
        )
        try:
            text = (self._generate or self._default_generate)(prompt).strip()
        except Exception:
            return facts
        return text or facts


def explain_all(shows: list[Show], songs: list[str], explainer: Explainer | None = None) -> list[str]:
    explainer = explainer or TemplateExplainer()
    return [explainer.explain(e) for e in gather_evidence(shows, songs)]
