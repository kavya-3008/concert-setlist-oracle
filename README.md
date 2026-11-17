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
| Per-slot (opener/closer) | 76% | 77% |
| Recency + city (k=3) | 73% | 78% |
| Recency, auto-tuned decay | 76% | 78% |

Recency helps because songs enter and leave rotation. Markov ordering hurts here since mid-set order is random in this data: a reminder to test on real tours before trusting a fancier model.

### Recalibration (isotonic)

The reliability table showed the recency model is underconfident for rare songs (predicted ~7%, played ~28%) and overconfident for staples (predicted ~98%, played ~92%). An isotonic calibrator (pool-adjacent-violators) fixes that shape. To avoid leaking, each held-out show is recalibrated using only earlier held-out folds, and the first 3 folds are skipped as warm-up (464 song-show cases scored).

| Probabilities | Brier score |
|---|---|
| Raw recency model | 0.2003 |
| Isotonic-recalibrated | 0.1975 |

That is only about a 1.4% improvement on 36 synthetic shows. The calibrator has little data to learn from, so treat it as a modest, unproven gain; it needs a real, longer tour to judge.

### Per-slot model (opener / closer)

`SlotModel` picks the opener and closer from recency-weighted slot frequencies, then fills the middle with the most probable remaining songs. Backtest on the synthetic tour (31 shows scored):

| Model | Opener exact | Closer exact | Songs recovered |
|---|---|---|---|
| Frequency | 100% | 100% | 72.7% |
| Recency-weighted | 100% | 100% | 75.5% |
| Per-slot | 100% | 100% | 75.5% |

Negative result: no gain. In this synthetic tour the opener and closer never change, so every model already gets them right, and the slot model reduces to the recency model. The test is not informative about real tours, where openers and closers rotate; the model is kept (with unit tests on a drifting closer) so it can be evaluated once real setlists are available.

### City effects

`CityModel` blends each song's recency probability with that song's rate in past shows in the target city, shrunk toward the global rate (`p = (n*p_city + k*p_global)/(n + k)`). `city_lift` lists the biggest city-vs-overall gaps. Backtest on the synthetic tour (31 shows scored), songs recovered: recency 75.5%; city model 71.8% (k=1), 73.4% (k=3), 73.9% (k=10).

Negative result: the city model is worse than plain recency at every shrinkage level, approaching it as k grows. The synthetic generator has no regional structure and each city has only 3-4 shows, so city rates mostly add noise (and ignore drift). Unit tests on a tour with injected East/West songs confirm the model does pick regional songs when the effect exists, so it is kept for real tours.

### Tuned recency decay

The recency model's decay (0.85) was a guess. `TunedRecencyModel` re-selects it at every fold from a grid (0.5 to 1.0) by running an inner walk-forward backtest on the training shows only, so nothing from the held-out show leaks in. A fixed-decay sweep on the full backtest (31 shows) gives songs recovered of 74.4% (0.5), 76.5% (0.6), 75.2% (0.7), 75.3% (0.8), 75.5% (0.85), 74.0% (0.9), 73.2% (0.95), 72.7% (1.0).

| Model | Songs recovered | Order accuracy |
|---|---|---|
| Recency (decay 0.85) | 75.5% | 76.5% |
| Recency, auto-tuned decay | 75.6% | 77.9% |

Negative result: tuning is a tie (+0.07 points of overlap, well within noise on 31 shows). The best fixed decay in hindsight (0.6) is about one point better, but the sweep is flat enough that a single show swings it, and the tuner mostly picked 1.0 early on (little drift visible yet) and 0.6 late. The tuner is kept because it removes a hand-set knob and should matter on real tours whose drift speed is unknown.

### Error analysis

`error_analysis` splits backtest misses into *unreachable* (song never played earlier in the tour, so no history-based model can name it) and *reachable* (seen before, but left out of the prediction), and lists the most-missed and most-wrongly-picked songs. On the synthetic tour (31 shows scored, overlap pooled over all played songs rather than averaged per show, so it differs slightly from the tables above):

| Model | Songs recovered | Ceiling (seen-only) | Misses that were unreachable |
|---|---|---|---|
| Frequency | 72.9% | 99.3% | 2.2% |
| Recency-weighted | 75.9% | 99.3% | 2.5% |

Finding: the ceiling is ~99%, so almost all of the ~24% miss is reachable rotation error, not unseen songs. The weakness is deciding which rotating songs play on a given night, not cold start. Both models keep missing and over-picking the same mid-rotation songs (e.g. Last Train Home, Glass Summer, Paper Planes), which are roughly coin flips in this generator. Nothing here beats the baseline; it is a diagnostic and points future modeling effort at the rotating slots, not at surprise songs.

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
- [x] Per-slot model: opener and closer predicted separately (ties the recency model on synthetic data; encore/surprise slots still open)
- [x] Recalibrate probabilities (isotonic regression, evaluated out-of-sample): small gain only
- [ ] Small transformer over song sequences, only if real data shows order structure
- [ ] Gemini-written explanations for each pick ("why this song?")
- [x] City effects: do tours rotate songs by region? (shrunk per-city model; loses to recency on synthetic data, which has no regional structure)
- [x] Tune the recency decay without leakage (inner walk-forward; ties the fixed 0.85 on synthetic data)
- [x] Error analysis: reachable vs unreachable misses, most-missed songs (shown on the site)
- [ ] Small web UI

## Config

Copy `.env.example` to `.env` and add `GEMINI_API_KEY`. The key is read from the environment and never committed.

## Live site

https://kavya-3008.github.io/concert-setlist-oracle/ (regenerate with `PYTHONPATH=. python3 scripts/build_site.py`)
