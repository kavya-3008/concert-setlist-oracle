# Concert Setlist Oracle

[![tests](https://github.com/kavya-3008/concert-setlist-oracle/actions/workflows/tests.yml/badge.svg)](https://github.com/kavya-3008/concert-setlist-oracle/actions/workflows/tests.yml) · **[Live demo](https://kavya-3008.github.io/concert-setlist-oracle/)**

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
| Recency + linear trend | 75% | 77% |
| Recency + carry-over (k=3) | 75% | 80% |

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

### Linear trend forecast

`TrendModel` asks whether a song's rate is *heading* somewhere rather than just where it has been: for each song it fits a decay-weighted straight line to its 0/1 presence over show index, forecasts one step ahead (clipped to [0, 1]) and blends that with the recency probability (`shrink` = share of trend). The synthetic tour phases songs in and out linearly, so this is the case where trend should help. Backtest (31 shows), songs recovered / order accuracy:

| Model | Songs recovered | Order accuracy |
|---|---|---|
| Recency (decay 0.85) | 75.5% | 76.5% |
| Trend, decay 0.9, shrink 0.5 (default) | 75.2% | 77.2% |
| Trend, decay 0.8, shrink 0.5 | 74.8% | 77.6% |
| Trend, decay 0.8, shrink 1.0 | 74.1% | 78.5% |
| Trend, decay 0.9, shrink 1.0 | 73.6% | 77.9% |
| Trend, decay 1.0, shrink 0.5 | 73.9% | 77.7% |
| Trend, decay 1.0, shrink 1.0 | 75.7% | 77.5% |

Negative result: no reliable gain. Across six settings song recovery ranges from 73.6% to 75.7% against 75.5% for plain recency, with no pattern in the settings (the single best one, +0.2 points, was picked in hindsight on 31 shows and is noise). Per-song 0/1 series are too noisy for a slope to be estimated from a few dozen shows, and the extrapolation mainly adds variance. Order accuracy moves up by about one point, but that comes from ranking probabilities differently, not from better song choice, so I would not read much into it. Kept as an informative baseline with unit tests for when a longer real tour is available.

### Carry-over from the previous show

`CarryoverModel` asks whether a song's chance depends on whether it was played at the previous show (negative for deliberate rotation, positive for songs being phased in). For each song it estimates P(play | played last show) and P(play | absent last show) from the history, shrunk toward the recency probability with strength `k`, and uses the one matching the last observed show. Backtest (31 shows), songs recovered / order accuracy:

| Model | Songs recovered | Order accuracy |
|---|---|---|
| Recency (decay 0.85) | 75.5% | 76.5% |
| Carry-over, k=1 | 72.9% | 80.4% |
| Carry-over, k=3 (default) | 74.5% | 80.3% |
| Carry-over, k=10 | 74.3% | 79.4% |
| Carry-over, k=30 | 74.4% | 77.3% |

Negative result: carry-over loses 1 to 2.6 points of song recovery at every shrinkage level, and the loss shrinks as k grows toward plain recency. A song's per-state rate rests on only a handful of transitions in a 36-show tour, so the estimate adds variance without a real signal to find. (Raw pooled rates look different, 72% after a play versus 45% after an absence, but that is mostly staples versus rare songs, not carry-over.) The order-accuracy rise of about 4 points comes from the model ranking probabilities differently, not from choosing better songs, so I would not read it as a win. Kept as an informative baseline with unit tests on alternating songs.

### Is any gain real? Paired bootstrap

Several sections above call differences "within noise" by eye. `oracle/significance.py` makes that explicit: each model's per-show song overlap is paired with recency's on the same 31 folds, and the shows are resampled 2000 times for a 95% interval on the mean difference (points of overlap, model minus recency).

| Model vs recency | Mean diff | 95% interval | Excludes 0? |
|---|---|---|---|
| Frequency | -2.8 | -5.7 to -0.3 | yes |
| Recency + Markov order | 0.0 | 0.0 to 0.0 | no (same songs) |
| Per-slot | 0.0 | 0.0 to 0.0 | no (same songs) |
| Recency + city (k=3) | -2.1 | -5.2 to +0.6 | no |
| Auto-tuned decay | +0.1 | -1.8 to +1.9 | no |
| Linear trend | -0.4 | -2.0 to +1.2 | no |
| Carry-over (k=3) | -1.0 | -4.2 to +2.0 | no |

Finding: the one reliable result is that recency beats plain frequency (about 2.8 points). Nothing else is distinguishable from recency, so the earlier "ties" and small losses are honestly undecided rather than established; with 31 shows an interval is roughly plus or minus 2 to 3 points, so only effects larger than that can be detected. Caveat: the interval treats shows as independent, but neighbouring shows share rotation state, so it is likely a little too narrow.

### Error analysis

`error_analysis` splits backtest misses into *unreachable* (song never played earlier in the tour, so no history-based model can name it) and *reachable* (seen before, but left out of the prediction), and lists the most-missed and most-wrongly-picked songs. On the synthetic tour (31 shows scored, overlap pooled over all played songs rather than averaged per show, so it differs slightly from the tables above):

| Model | Songs recovered | Ceiling (seen-only) | Misses that were unreachable |
|---|---|---|---|
| Frequency | 72.9% | 99.3% | 2.2% |
| Recency-weighted | 75.9% | 99.3% | 2.5% |

Finding: the ceiling is ~99%, so almost all of the ~24% miss is reachable rotation error, not unseen songs. The weakness is deciding which rotating songs play on a given night, not cold start. Both models keep missing and over-picking the same mid-rotation songs (e.g. Last Train Home, Glass Summer, Paper Planes), which are roughly coin flips in this generator. Nothing here beats the baseline; it is a diagnostic and points future modeling effort at the rotating slots, not at surprise songs.

### Set-length forecasting

Every backtest above is handed the true set length, which nobody has before the show. `oracle/length.py` forecasts it from past shows and scores the whole pipeline (forecast length, then predict that many recency-model songs) with precision, recall and F1. In the synthetic tour sets grow from about 9-11 songs early to 10-13 late. Backtest, 31 shows:

| Length forecast | Mean abs. error (songs) | Bias | F1 |
|---|---|---|---|
| Overall mean | 1.10 | -0.58 | 73.7% |
| Decayed mean (0.85) | 0.97 | -0.19 | 75.1% |
| Last 5 shows | 1.10 | -0.13 | 75.5% |
| (reference: true length given) | 0 | 0 | 75.5% |

Result: a recency-aware length forecast recovers nearly all of the loss from not knowing the length (F1 75.1-75.5% vs 75.5% with the true length), while the plain mean, the current CLI default, under-predicts by about half a song and costs about 1.8 points of F1. Caveats: the gaps between the two recency-aware forecasters are within noise on 31 shows, a decay sweep (0.5 to 0.9) moved F1 only between 74.9% and 75.7%, and mean absolute error is not better for the last-5 forecaster than the overall mean (the gain is in bias, which is what shifts precision against recall). The sum of the recency model's probabilities is mathematically identical to the decayed mean length, so it is not a separate row. Also, F1 with a true length equals the overlap metric used elsewhere, so the real finding is that length error costs under 2 points here.

### Surprise candidates, backtested

The "surprise candidates" feature (rotating songs that are "due") had never been evaluated. `oracle/surprise.py` scores any ranking rule by precision@k: of the k rotating songs (play rate 5-50%) it names before a show, how many were actually played, versus the base rate of a random pick from the same pool. Backtest, 31 shows, k=5 (k=3 in brackets):

| Ranking rule | Top-k played | Random pool pick |
|---|---|---|
| Due heuristic (recent rate minus overall rate, the old default) | 44.7% (44.1%) | 48.0% |
| Longest gap since last played | 46.7% (43.0%) | 48.0% |
| Recency-weighted probability | 54.6% (54.8%) | 48.0% |

Finding: the original "due" heuristic is slightly worse than picking at random, and a gap-based rule is no better. In this synthetic tour songs are phased in and out rather than on a cycle, so being absent signals that a song is leaving, not that it is overdue. Ranking by recency probability beats the base rate by about 6.5 points at both k. That is only 155 picks, so the standard error is roughly 4 points: suggestive, not conclusive. The site now lists surprise candidates by the recency rule; `SetlistModel.surprise_candidates` is left unchanged for comparison.

### Explanations ("why this song?")

`oracle/explain.py` gathers per-pick evidence from the shows (play count, recent appearances, shows since last played, recency-weighted probability) and renders it as a sentence. `TemplateExplainer` works offline and is what the site uses. `GeminiExplainer` asks an LLM to rephrase the same facts, takes an injectable `generate` callable, and falls back to the template when there is no `GEMINI_API_KEY` or the call fails. Because the numbers come from code, not from the model, an explanation cannot invent statistics.

This changes no predictions, so there is nothing to backtest: it is a presentation feature. The Gemini path is only unit-tested against a mock and has not been run against the real API.

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
- [x] Trend forecast: extrapolate each song's rate with a weighted line (ties recency on synthetic data)
- [ ] Small transformer over song sequences, only if real data shows order structure
- [x] Explanations for each pick ("why this song?"): evidence computed from the shows, offline template explainer, optional Gemini rephrasing behind an injectable interface (tested with a mock; no live Gemini call has been made)
- [x] City effects: do tours rotate songs by region? (shrunk per-city model; loses to recency on synthetic data, which has no regional structure)
- [x] Tune the recency decay without leakage (inner walk-forward; ties the fixed 0.85 on synthetic data)
- [x] Error analysis: reachable vs unreachable misses, most-missed songs (shown on the site)
- [x] Set-length forecasting: predict how many songs, score the full pipeline with F1 (recency-aware forecast nearly matches knowing the true length; shown on the site)
- [x] Backtest the surprise-candidate feature (the old "due" rule does not beat random; recency ranking does by ~6.5 points, within noise-ish; shown on the site)
- [x] Carry-over model: does playing a song last night change tonight's odds? (loses to recency on synthetic data; shown on the site)
- [x] Paired-bootstrap significance of every model vs recency (only recency > frequency is reliable; shown on the site)
- [ ] Small web UI

## Config

Copy `.env.example` to `.env` and add `GEMINI_API_KEY`. The key is read from the environment and never committed.

## Live site

https://kavya-3008.github.io/concert-setlist-oracle/ (regenerate with `PYTHONPATH=. python3 scripts/build_site.py`)
