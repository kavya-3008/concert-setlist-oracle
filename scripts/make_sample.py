"""Generate a synthetic tour by a fictional band, with realistic drift. Run: python3 scripts/make_sample.py"""
import json
import random
from pathlib import Path

random.seed(11)
staples = ["Neon Tides", "Paper Moons", "Glass Summer", "Static Hearts", "Weekend Ghosts", "Slow Parade", "Last Train Home"]
early = ["Cassette Hymn", "Lighthouse", "Blue Tarmac"]  # fade out as the tour goes on
late = ["Velvet Alarm", "Salt & Smoke", "Paper Planes"]  # new songs worked in later
cities = ["San Diego", "Los Angeles", "Phoenix", "Austin", "Denver", "Seattle", "Portland", "Chicago", "Boston", "New York"]
N = 36
shows = []
for i in range(N):
    t = i / (N - 1)
    mid = random.sample(staples, 5)
    pool = [(s, 1 - t) for s in early] + [(s, t) for s in late]
    mid += [s for s, p in pool if random.random() < 0.15 + 0.7 * p]
    random.shuffle(mid)
    if i % 9 == 8:
        mid = mid[:-1] + ["Golden Hour Radio"]  # an occasional surprise cover
    songs = ["Open Windows"] + mid + (["Encore Lights"] if t > 0.5 else []) + ["Everyone Sings"]
    shows.append({"date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", "city": cities[i % len(cities)], "songs": songs})
Path(__file__).resolve().parent.parent.joinpath("data/sample_tour.json").write_text(json.dumps(shows, indent=1))
print(f"wrote {len(shows)} shows")
