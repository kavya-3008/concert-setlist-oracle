"""Regenerate docs/index.html (the GitHub Pages site) from the sample tour. Run: PYTHONPATH=. python3 scripts/build_site.py"""
from html import escape
from pathlib import Path

from oracle.data import load_shows
from oracle.evaluate import backtest
from oracle.model import SetlistModel

ROOT = Path(__file__).resolve().parent.parent
shows = load_shows(ROOT / "data/sample_tour.json")
model = SetlistModel().fit(shows)
bt = backtest(shows)

setlist = "".join(f"<li>{escape(s)}</li>" for s in model.predict())
surprise = "".join(f"<li>{escape(s)}</li>" for s in model.surprise_candidates())
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
<div class="card" style="margin-top:16px"><h2 style="margin-top:0">Backtest</h2><div class="stat">{bt['mean_overlap']:.0%}</div>
<p class="note" style="margin:0">of each real setlist recovered, predicting {bt['shows_scored']} shows using only the shows before each one.</p></div></div></div>
<h2>Every song</h2><div class="card"><table><tr><th>Song</th><th>Play rate</th><th></th><th>Set position</th></tr>{rows}</table>
<p class="note">Set position: 0 = opener, 1 = closer.</p></div></main></body></html>"""
(ROOT / "docs/index.html").write_text(html)
print("wrote docs/index.html")
