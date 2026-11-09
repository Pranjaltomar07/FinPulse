import sys, json, sqlite3, ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent
B, O, G, INK, MUT = "#2a78d6", "#eb6834", "#9a9a95", "#1a1a19", "#6b6a66"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
  "axes.edgecolor": "#bbb", "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.titleweight": "bold",
  "axes.titlelocation": "left", "axes.titlesize": 13, "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": True,
  "grid.color": "#e8e8e4", "grid.linewidth": .7, "axes.axisbelow": True})
def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); print(p)

# ---- FinPulse
f = R; img = f/"docs"/"img"
fe = pd.read_csv(f/"reports/forecast_eval.csv"); t = fe[fe.category == "TOTAL"].iloc[0]
fig, ax = plt.subplots(figsize=(6.4, 4.2)); names = ["Random forest", "Naive (last month)", "3-month moving avg"]; vals = [t.rf_wape, t.naive_last_month_wape, t.moving_avg3_wape]
ax.bar(names, vals, color=[O, G, B], width=.55)
for i, v in enumerate(vals): ax.text(i, v+.4, f"{v:.1f}%", ha="center", color=INK)
ax.set_ylabel("WAPE, total monthly spend (lower is better)"); ax.set_title("Simple average beats the ML forecast"); ax.grid(axis="x", visible=False)
ax.text(.99, .95, "Simulated data", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUT); save(fig, img/"forecast_wape.png")
