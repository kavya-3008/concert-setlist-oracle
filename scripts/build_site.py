"""Regenerate docs/index.html (the GitHub Pages site) from the sample tour. Run: PYTHONPATH=. python3 scripts/build_site.py"""
from html import escape
from pathlib import Path

from oracle.carryover import CarryoverModel
from oracle.city import CityModel
from oracle.data import load_shows
from oracle.calibration import brier, probability_pairs, recalibrated_pairs, reliability
from oracle.errors import error_analysis
from oracle.explain import gather_evidence, TemplateExplainer
from oracle.evaluate import compare
from oracle.length import LENGTH_FORECASTERS, length_backtest
from oracle.markov import MarkovModel
from oracle.model import SetlistModel
from oracle.recency import RecencyModel
from oracle.slots import SlotModel
from oracle.surprise import RANKERS, recency_ranking, surprise_backtest
from oracle.trend import TrendModel
from oracle.tuning import TunedRecencyModel

ROOT = Path(__file__).resolve().parent.parent
shows = load_shows(ROOT / "data/sample_tour.json")
model = MarkovModel().fit(shows)
res = compare(shows, {'Frequency': SetlistModel, 'Recency-weighted': RecencyModel, 'Recency + Markov order': MarkovModel, 'Per-slot (opener/closer)': SlotModel, 'Recency + city (k=3)': CityModel, 'Recency, auto-tuned decay': TunedRecencyModel, 'Recency + linear trend': TrendModel, 'Recency + carry-over (k=3)': CarryoverModel})
pairs = list(probability_pairs(shows))
br = brier(pairs)
rel = reliability(pairs)
raw_cv, cal_cv = recalibrated_pairs(shows)
br_raw, br_cal = brier(raw_cv), brier(cal_cv)
len_rows = ''.join(f"<tr><td>{escape(n)}</td><td>{r['mae']:.2f}</td><td>{r['bias']:+.2f}</td><td>{r['f1']:.1%}</td></tr>" for n, r in ((n, length_backtest(shows, f)) for n, f in LENGTH_FORECASTERS.items()))
ea = error_analysis(shows, RecencyModel)
miss_rows = ''.join(f"<tr><td>{escape(s)}</td><td>{n}</td></tr>" for s, n in ea['most_missed'])
spur_rows = ''.join(f"<tr><td>{escape(s)}</td><td>{n}</td></tr>" for s, n in ea['most_spurious'])
cmp_rows = ''.join(f"<tr><td>{escape(n)}</td><td>{r['overlap']:.0%}</td><td>{r['order']:.0%}</td></tr>" for n, r in res.items())
rel_rows = ''.join(f"<tr><td>{b['bin']}</td><td>{b['predicted']:.0%}</td><td>{b['observed']:.0%}</td><td>{b['n']}</td></tr>" for b in rel)

setlist = "".join(f"<li>{escape(s)}</li>" for s in model.predict())
surprise = "".join(f"<li>{escape(s)}</li>" for s in recency_ranking(shows, 0.05, 0.5)[:5])
sur_rows = ''.join(f"<tr><td>{escape(n)}</td><td>{r['precision']:.0%}</td><td>{r['base_rate']:.0%}</td></tr>" for n, r in ((n, surprise_backtest(shows, f)) for n, f in RANKERS.items()))
why_rows = ''.join(f"<tr><td>{escape(e.song)}</td><td>{e.probability:.0%}</td><td>{escape(TemplateExplainer().explain(e).split(': ', 1)[1])}</td></tr>" for e in gather_evidence(shows, model.predict()))
rows = "".join(
    f'<tr><td>{escape(s.song)}</td><td><div class="bar"><span style="width:{s.play_rate*100:.0f}%"></span></div></td>'
    f"<td>{s.play_rate:.0%}</td><td>{s.mean_pos:.2f}</td></tr>"
    for s in sorted(model.stats.values(), key=lambda s: -s.play_rate)
)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Concert Setlist Oracle</title><style>
:root{{--bg:#fbf8f3;--fg:#1d1b19;--mut:#6b655d;--acc:#c2410c;--card:#fff;--line:#e8e1d6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16130f;--fg:#f3ede4;--mut:#a39a8c;--acc:#fb923c;--card:#1f1b16;--line:#33291f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:40px 16px 72px}}h1{{font-size:2.2rem;margin:0 0 6px}}h2{{margin:36px 0 12px}}
.lede{{color:var(--mut);max-width:60ch}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 22px}}
ol,ul{{margin:0;padding-left:22px}}li{{padding:2px 0}}.stat{{font-size:2.4rem;font-weight:700;color:var(--acc)}}
table{{width:100%;border-collapse:collapse;font-size:.92rem}}td,th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}}
.bar{{height:8px;background:var(--line);border-radius:4px;min-width:120px}}.bar span{{display:block;height:100%;background:var(--acc);border-radius:4px}}
a{{color:var(--acc)}}.note{{color:var(--mut);font-size:.88rem}}</style></head><body><main>
<h1>Concert Setlist Oracle</h1>
<p class="lede">Tours are more regular than they look: a core set every night, plus a few rotating slots. This predicts the next setlist from past shows and flags which songs are "due" as a surprise.</p>
<p class="note">Demo data is a synthetic tour by a fictional band ({len(shows)} shows). <a href="https://github.com/kavya-3008/concert-setlist-oracle">Source on GitHub</a>.</p>
<div class="grid"><div class="card"><h2 style="margin-top:0">Predicted setlist</h2><ol>{setlist}</ol></div>
<div><div class="card"><h2 style="margin-top:0">Surprise candidates</h2><ul>{surprise}</ul></div>
<div class="card" style="margin-top:16px"><h2 style="margin-top:0">Best backtest</h2><div class="stat">{max(r['overlap'] for r in res.values()):.0%}</div>
<p class="note" style="margin:0">of each real setlist recovered, predicting every show from only the shows before it.</p></div></div></div>
<h2>Why these picks?</h2><div class="card"><table><tr><th>Song</th><th>Chance</th><th>Evidence</th></tr>{why_rows}</table>
<p class="note">Written from the tour's own counts. An optional Gemini explainer can rephrase the same evidence when a key is set; it is not used on this page.</p></div>
<h2>Model comparison</h2><div class="card"><table><tr><th>Model</th><th>Songs recovered</th><th>Order accuracy</th></tr>{cmp_rows}</table>
<p class="note">Order accuracy: of the songs both lists share, how many pairs are in the right relative order. Markov ordering does worse here because mid-set order in this synthetic tour is random; real tours should differ.</p></div>
<h2>How long will the show be?</h2><div class="card"><table><tr><th>Length forecast</th><th>Avg error (songs)</th><th>Bias</th><th>F1 of predicted set</th></tr>{len_rows}</table>
<p class="note">Other tables are told the true set length; here the length is forecast first. Recent-weighted forecasts track the synthetic tour's growing sets; the overall mean runs short.</p></div>
<h2>Where do the misses come from?</h2><div class="card"><p style="margin-top:0">The recency model recovers {ea['overlap']:.0%} of played songs (pooled over {ea['shows_scored']} shows). Only songs never played earlier are out of reach, and that caps recovery at {ea['ceiling']:.0%}; just {ea['unreachable_share']:.0%} of misses are of that kind. The rest are rotation errors: songs it knew about but left out.</p>
<div class="grid"><div><table><tr><th>Most often missed</th><th>Shows</th></tr>{miss_rows}</table></div><div><table><tr><th>Most often wrongly picked</th><th>Shows</th></tr>{spur_rows}</table></div></div></div>
<h2>Do the surprise picks land?</h2><div class="card"><table><tr><th>Ranking rule</th><th>Top-5 actually played</th><th>Random rotating song</th></tr>{sur_rows}</table>
<p class="note">Each rule picks 5 rotating songs per show from earlier shows only. The old "due" rule did not beat a random pick; ranking by recent play probability did, by a few points on 31 synthetic shows. The list above uses that rule.</p></div>
<h2>Are the probabilities honest?</h2><div class="card"><table><tr><th>Predicted bin</th><th>Avg predicted</th><th>Actually played</th><th>Cases</th></tr>{rel_rows}</table>
<p class="note">Brier score: {br:.3f} (lower is better). Well-calibrated means the two percentage columns match. Isotonic recalibration (fit only on earlier shows) moves Brier from {br_raw:.4f} to {br_cal:.4f} on the same cases after a 3-fold warm-up: a small gain on this synthetic tour.</p></div>
<h2>Every song</h2><div class="card"><table><tr><th>Song</th><th>Play rate</th><th></th><th>Set position</th></tr>{rows}</table>
<p class="note">Set position: 0 = opener, 1 = closer.</p></div></main></body></html>"""
(ROOT / "docs/index.html").write_text(html)
print("wrote docs/index.html")
