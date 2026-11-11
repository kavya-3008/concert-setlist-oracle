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

## Findings so far (synthetic tour with drift)

| Model | Songs recovered | Order accuracy |
|---|---|---|
| Frequency | 73% | 77% |
| Recency-weighted | 76% | 77% |
| Recency + Markov order | 76% | 67% |

Recency helps because songs enter and leave rotation. Markov ordering hurts here since mid-set order is random in this data: a reminder to test on real tours before trusting a fancier model.

### Recalibration (isotonic)

The reliability table showed the recency model is underconfident for rare songs (predicted ~7%, played ~28%) and overconfident for staples (predicted ~98%, played ~92%). An isotonic calibrator (pool-adjacent-violators) fixes that shape. To avoid leaking, each held-out show is recalibrated using only earlier held-out folds, and the first 3 folds are skipped as warm-up (464 song-show cases scored).

| Probabilities | Brier score |
|---|---|
| Raw recency model | 0.2003 |
| Isotonic-recalibrated | 0.1975 |

That is only about a 1.4% improvement on 36 synthetic shows. The calibrator has little data to learn from, so treat it as a modest, unproven gain; it needs a real, longer tour to judge.

## How it works (v0.1)

- **Play rate**: fraction of shows a song appears in. High means a staple.
- **Typical position**: average normalized slot in the set (0 = opener, 1 = closer).
- **Prediction**: take the most likely songs for the typical set length, ordered by typical position.
- **Surprise candidates**: songs that are played sometimes but have been absent recently, so they are "due".
- **Backtest**: predict each show from only the shows before it and report how much of the real setlist was recovered.

## Roadmap

- [x] Baseline frequency/position model + backtest
- [ ] Fetch real tours from the setlist.fm API
- [x] Recency-weighted probabilities (tours drift)
- [x] Markov ordering model, compared in the backtest (does not beat the baseline on synthetic data)
- [x] Calibration: reliability bins and Brier score
- [ ] Per-slot model (opener, closer, encore, "surprise slot")
- [x] Recalibrate probabilities (isotonic regression, evaluated out-of-sample): small gain only
- [ ] Small transformer over song sequences, only if real data shows order structure
- [ ] Gemini-written explanations for each pick ("why this song?")
- [ ] City effects: do tours rotate songs by region?
- [ ] Small web UI

## Config

Copy `.env.example` to `.env` and add `GEMINI_API_KEY`. The key is read from the environment and never committed.

## Live site

https://kavya-3008.github.io/concert-setlist-oracle/ (regenerate with `PYTHONPATH=. python3 scripts/build_site.py`)
