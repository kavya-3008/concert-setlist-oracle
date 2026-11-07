# Concert Setlist Oracle

Predict tonight's setlist from a tour's past shows, and guess which "surprise song" might show up.

Tours are surprisingly regular: most artists play a core set every night and rotate a handful of slots.
This project models that structure, starting with a transparent baseline and growing toward sequence models and LLM-written explanations.

## Quickstart

```bash
python3 -m pip install -r requirements.txt
python3 -m oracle.cli data/sample_tour.json
python3 -m pytest
```

`data/sample_tour.json` is **synthetic** (a fictional band) so the project runs offline.

## How it works (v0.1)

- **Play rate**: fraction of shows a song appears in. High means a staple.
- **Typical position**: average normalized slot in the set (0 = opener, 1 = closer).
- **Prediction**: take the most likely songs for the typical set length, ordered by typical position.
- **Surprise candidates**: songs that are played sometimes but have been absent recently, so they are "due".
- **Backtest**: predict each show from only the shows before it and report how much of the real setlist was recovered.

## Roadmap

- [x] Baseline frequency/position model + backtest
- [ ] Fetch real tours from the setlist.fm API
- [ ] Per-slot model (opener, closer, encore, "surprise slot")
- [ ] Sequence model (Markov / small transformer) and compare against the baseline in the backtest
- [ ] Calibrated probabilities per song, with reliability plots
- [ ] Gemini-written explanations for each pick ("why this song?")
- [ ] City effects: do tours rotate songs by region?
- [ ] Small web UI

## Config

Copy `.env.example` to `.env` and add `GEMINI_API_KEY`. The key is read from the environment and never committed.

## Live site

https://kavya-3008.github.io/concert-setlist-oracle/ (regenerate with `PYTHONPATH=. python3 scripts/build_site.py`)
