"""Usage: python -m oracle.cli data/sample_tour.json"""
from __future__ import annotations

import argparse

from .data import load_shows
from .evaluate import backtest, compare
from .markov import MarkovModel
from .model import SetlistModel
from .recency import RecencyModel


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict the next setlist for a tour")
    ap.add_argument("data", help="path to a tour JSON file")
    args = ap.parse_args()

    shows = load_shows(args.data)
    model = MarkovModel().fit(shows)

    print(f"Trained on {len(shows)} shows.\n\nPredicted setlist:")
    for i, song in enumerate(model.predict(), 1):
        print(f"  {i:>2}. {song}")
    print("\nSurprise song candidates:", ", ".join(model.surprise_candidates()) or "none")
    if len(shows) > 5:
        res = compare(shows, {"frequency": SetlistModel, "recency": RecencyModel, "markov": MarkovModel})
        print("\nBacktest (rolling, predicts each show from the ones before it):")
        for name, r in res.items():
            print(f"  {name:<10} songs recovered {r['overlap']:.0%}   order accuracy {r['order']:.0%}")


if __name__ == "__main__":
    main()
