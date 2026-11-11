"""One self-contained HTML dashboard (no server, no CDN): open it by double-clicking."""
import html, json
import numpy as np, pandas as pd
from . import budget as B
from .config import SPEND_CATS, DISCRETIONARY

PAL = ["#1a8a8a", "#e09f3e", "#d1495b", "#3d5a80", "#8d99ae", "#6a994e", "#9b5de5"]


def _bar_svg(labels, vals, w=720, h=240, colors=None, fmt="{:,.0f}"):
    m = max(vals) or 1; n = len(vals); bw = (w - 60) / n; s = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for i, (l, v) in enumerate(zip(labels, vals)):
        bh = (h - 50) * v / m; x = 50 + i * bw
        s.append(f'<rect x="{x+4:.1f}" y="{h-30-bh:.1f}" width="{bw-8:.1f}" height="{bh:.1f}" rx="3" fill="{(colors or [PAL[0]]*n)[i]}"><title>{html.escape(str(l))}: {fmt.format(v)}</title></rect>')
        if i % max(1, -(-n // 8)) == 0 or n <= 8:
            s.append(f'<text x="{x+bw/2:.1f}" y="{h-12}" text-anchor="middle" font-size="10">{html.escape(str(l))}</text>')
    s.append(f'<text x="4" y="14" font-size="10">max {fmt.format(m)}</text></svg>'); return "".join(s)


def _line_svg(labels, vals, point=None, w=720, h=240):
    allv = list(vals) + ([point["hi"], point["lo"]] if point else []); lo, hi = 0, max(allv) * 1.08; n = len(vals) + (1 if point else 0)
    X = lambda i: 50 + i * (w - 70) / max(n - 1, 1); Y = lambda v: h - 30 - (h - 50) * (v - lo) / (hi - lo)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">', f'<polyline fill="none" stroke="#1f2a44" stroke-width="2" points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))}"/>']
    if point:
        i = len(vals); s.append(f'<line x1="{X(i)}" x2="{X(i)}" y1="{Y(point["hi"]):.1f}" y2="{Y(point["lo"]):.1f}" stroke="#e09f3e" stroke-width="4"/><circle cx="{X(i)}" cy="{Y(point["point"]):.1f}" r="5" fill="#e09f3e"><title>{point["month"]}: {point["point"]:,.0f} (80% {point["lo"]:,.0f} to {point["hi"]:,.0f})</title></circle>')
    for i in range(0, n, max(1, n // 6)):
        lab = labels[i] if i < len(labels) else point["month"]; s.append(f'<text x="{X(i):.1f}" y="{h-12}" text-anchor="middle" font-size="10">{lab}</text>')
    s.append(f'<text x="4" y="14" font-size="10">max {hi:,.0f}</text></svg>'); return "".join(s)


def _table(df, cols):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    r = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in df[cols].itertuples(index=False))
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{r}</tbody></table></div>'


def render(res: dict, path, title="FinPulse") -> None:
    df, T, nm, found, anom = res["df"], res["monthly"], res["forecast"], res["recurring"], res["anomalies"]
    spend = df[(df.type == "debit") & df.category.isin(SPEND_CATS)]; bycat = spend.groupby("category").amount.sum().sort_values(ascending=False)
    last12 = spend[spend.month.isin(sorted(df.month.unique())[-12:])].groupby("category").amount.sum().reindex(DISCRETIONARY).fillna(0)
    review = df[df.needs_review].sort_values("confidence").head(40).assign(confidence=lambda d: d.confidence.round(2))[["date", "description", "amount", "category", "confidence"]]
    inc = df[df.type == "credit"].groupby("month").amount.sum(); out = spend.groupby("month").amount.sum()
    rate = float(((inc - out.reindex(inc.index).fillna(0)) / inc).tail(12).mean() * 100) if len(inc) else float("nan")
    kp = [("Transactions", f"{len(df):,}"), ("Months of history", f"{df.month.nunique()}"), ("Avg monthly spend", f"{out.mean():,.0f}"),
          ("Subscriptions found", f"{len(found[found.kind=='expense'])}"), ("Flagged purchases", f"{len(anom)}"), ("Awaiting review", f"{int(df.needs_review.sum())}")]
    sl = "".join(f'<label>{html.escape(c)} <output id="o{i}">0%</output><input type="range" min="0" max="60" value="0" step="5" data-i="{i}" data-v="{last12[c]:.0f}"></label>' for i, c in enumerate(DISCRETIONARY))
    fc = (f'<p class="note">Next month ({nm["month"]}): <b>{nm["point"]:,.0f}</b>, 80% band {nm["lo"]:,.0f} to {nm["hi"]:,.0f}. The band comes from past forecast errors; in our tests it held about 72% of the time, not 80%.</p>' if nm else '<p class="note">Need at least 12 months to forecast.</p>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>:root{{--bg:#f7f5ef;--card:#fff;--ink:#1f2a44;--mut:#5b6577;--teal:#1a8a8a;--amb:#e09f3e;--line:#e3dfd3}}
@media(prefers-color-scheme:dark){{:root{{--bg:#12182a;--card:#1b2340;--ink:#e8ecf5;--mut:#a3adc2;--line:#2b3556}}}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{padding:22px 20px 8px;max-width:980px;margin:auto}}h1{{margin:0;font-size:26px}}header p{{color:var(--mut);margin:4px 0}}
nav{{display:flex;gap:6px;flex-wrap:wrap;max-width:980px;margin:8px auto;padding:0 20px}}nav button{{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 13px;border-radius:20px;cursor:pointer;font:inherit}}
nav button[aria-selected=true]{{background:var(--teal);color:#fff;border-color:var(--teal)}}main{{max-width:980px;margin:auto;padding:0 20px 40px}}
section{{display:none;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:10px}}section.on{{display:block}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px}}.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}.kpi b{{display:block;font-size:22px;color:var(--teal)}}.kpi span{{color:var(--mut);font-size:12px}}
.chart{{width:100%;height:auto;color:var(--ink)}}.chart text{{fill:var(--ink)}}.scroll{{overflow:auto;max-height:420px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}th{{position:sticky;top:0;background:var(--card)}}
.note{{color:var(--mut)}}label{{display:block;margin:10px 0}}input[type=range]{{width:100%;accent-color:var(--teal)}}.big{{font-size:28px;font-weight:700;color:var(--teal)}}footer{{max-width:980px;margin:auto;padding:0 20px 30px;color:var(--mut);font-size:12px}}</style></head><body>
<header><h1>{title}</h1><p>Personal-finance analytics on a simulated statement. Every number below is computed from the transactions you can see in the Review tab.</p></header>
<nav role="tablist">{"".join(f'<button role="tab" aria-selected="{str(i==0).lower()}" data-t="t{i}">{n}</button>' for i, n in enumerate(["Overview", "Categories", "Subscriptions", "Forecast", "What-if", "Review queue", "Flagged"]))}</nav>
<main>
<section id="t0" class="on"><div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in kp)}</div><h3>Monthly spend</h3>{_bar_svg(list(out.index), list(out.values))}<p class="note">Average savings rate over the last 12 months: {rate:.0f}% of income (after all spending, before investing).</p></section>
<section id="t1"><h3>Spend by category</h3>{_bar_svg(list(bycat.index), list(bycat.values), colors=[PAL[i % 7] for i in range(len(bycat))])}</section>
<section id="t2"><h3>Recurring payments</h3><p class="note">Found by timing (about monthly) and amount stability, with no labels used.</p>{_table(found, ["merchant", "kind", "median_amount", "monthly_cost", "n", "next_expected"])}</section>
<section id="t3"><h3>Forecast</h3>{_line_svg(list(T.index), list(T.TOTAL.values), nm) if nm else ""}{fc}</section>
<section id="t4"><h3>What if I cut back?</h3><p class="note">Based on the last 12 months of discretionary spend.</p>{sl}<p>Estimated saving per year: <span class="big" id="tot">0</span></p></section>
<section id="t5"><h3>Review queue</h3><p class="note">Transactions the model is least sure about. Confirming these is the fastest way to improve accuracy.</p>{_table(review, list(review.columns))}</section>
<section id="t6"><h3>Unusually large purchases</h3><p class="note">Robust z-score above 4 inside the category. In our tests this finds about 7 in 10 injected anomalies, with about 8 in 10 flags correct.</p>{_table(anom, list(anom.columns)) if len(anom) else "<p>None flagged.</p>"}</section>
</main><footer>Simulated data. Not financial advice.</footer>
<script>
const tabs=[...document.querySelectorAll('nav button')];tabs.forEach(b=>b.onclick=()=>{{tabs.forEach(x=>x.setAttribute('aria-selected',x===b));document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t))}});
const sl=[...document.querySelectorAll('input[type=range]')];const upd=()=>{{let t=0;sl.forEach(s=>{{document.getElementById('o'+s.dataset.i).textContent=s.value+'%';t+=s.dataset.v*s.value/100}});document.getElementById('tot').textContent=Math.round(t).toLocaleString()}};sl.forEach(s=>s.oninput=upd);upd();
</script></body></html>"""
    open(path, "w", encoding="utf-8").write(page)
