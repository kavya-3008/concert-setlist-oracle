"""A transparent baseline model: per-song play rate and typical set position."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .data import Show


@dataclass
class SongStats:
    song: str
    play_rate: float  # fraction of shows where the song was played
    mean_pos: float  # average normalized position in the set, 0 = opener, 1 = closer
    recent_rate: float  # play rate over the most recent shows (captures rotation)


class SetlistModel:
    def __init__(self, recent_window: int = 5):
        self.recent_window = recent_window
        self.stats: dict[str, SongStats] = {}
        self.avg_len: float = 0.0

    def fit(self, shows: list[Show]) -> "SetlistModel":
        if not shows:
            raise ValueError("need at least one show")
        n = len(shows)
        recent = shows[-self.recent_window :]
        played = defaultdict(int)
        positions = defaultdict(list)
        recent_played = defaultdict(int)
        for show in shows:
            length = len(show.songs)
            for i, song in enumerate(show.songs):
                played[song] += 1
                positions[song].append(i / max(length - 1, 1))
        for show in recent:
            for song in set(show.songs):
                recent_played[song] += 1
        self.avg_len = sum(len(s.songs) for s in shows) / n
        self.stats = {
            song: SongStats(
                song,
                played[song] / n,
                sum(positions[song]) / len(positions[song]),
                recent_played[song] / len(recent),
            )
            for song in played
        }
        return self

    def predict(self, length: int | None = None) -> list[str]:
        """Predict an ordered setlist: take the most likely songs, then order by typical position."""
        length = length or round(self.avg_len)
        ranked = sorted(self.stats.values(), key=lambda s: (-s.play_rate, s.song))
        chosen = ranked[:length]
        return [s.song for s in sorted(chosen, key=lambda s: (s.mean_pos, s.song))]

    def surprise_candidates(self, low: float = 0.05, high: float = 0.5, k: int = 5) -> list[str]:
        """Rotating songs: played sometimes, but not a staple. Ranked by how recently they've been absent."""
        pool = [s for s in self.stats.values() if low <= s.play_rate <= high]
        # Songs that are rare lately but not rare overall are "due".
        pool.sort(key=lambda s: (s.recent_rate - s.play_rate, s.song))
        return [s.song for s in pool[:k]]
