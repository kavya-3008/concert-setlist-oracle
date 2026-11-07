"""Setlist data structures and loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Show:
    date: str  # ISO date, e.g. "2026-03-14"
    city: str
    songs: tuple[str, ...]  # in played order


def load_shows(path: str | Path) -> list[Show]:
    """Load shows from a JSON file: [{"date", "city", "songs": [...]}, ...]. Sorted by date."""
    raw = json.loads(Path(path).read_text())
    shows = [Show(r["date"], r["city"], tuple(r["songs"])) for r in raw]
    return sorted(shows, key=lambda s: s.date)
