"""Usage: python -m oracle.cli data/sample_tour.json"""
from __future__ import annotations

import argparse

from .data import load_shows
from .evaluate import backtest
from .model import SetlistModel


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict the next setlist for a tour")
    ap.add_argument("data", help="path to a tour JSON file")
    args = ap.parse_args()

    shows = load_shows(args.data)
    model = SetlistModel().fit(shows)

    print(f"Trained on {len(shows)} shows.\n\nPredicted setlist:")
    for i, song in enumerate(model.predict(), 1):
        print(f"  {i:>2}. {song}")
    print("\nSurprise song candidates:", ", ".join(model.surprise_candidates()) or "none")
    if len(shows) > 5:
        r = backtest(shows)
        print(f"\nBacktest over {r['shows_scored']} shows: {r['mean_overlap']:.0%} of each real setlist predicted")


if __name__ == "__main__":
    main()
