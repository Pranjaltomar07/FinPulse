"""Static figures for the README and report."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

NAVY, TEAL, AMBER, CORAL, GREY = "#1f2a44", "#1a8a8a", "#e09f3e", "#d1495b", "#8d99ae"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "figure.dpi": 100, "savefig.dpi": 160, "axes.edgecolor": "#555"})


def _g(S, m): r = S[S.metric == m].iloc[0]; return r["mean"], r.ci_lo, r.ci_hi


def bars(ax, S, items, colors, pct=True, ylim=None, title=""):
    k = 100 if pct else 1
    vals = [_g(S, m) for m, _ in items]; x = np.arange(len(items))
    ax.bar(x, [v[0] * k for v in vals], color=colors, width=.62,
           yerr=[[(v[0] - v[1]) * k for v in vals], [(v[2] - v[0]) * k for v in vals]], capsize=3, ecolor="#333")
    for i, v in enumerate(vals): ax.text(i, v[2] * k + (1 if pct else .4), f"{v[0]*k:.0f}" if pct else f"{v[0]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in items]); ax.set_title(title, loc="left")
    if ylim: ax.set_ylim(*ylim)


def make_all(res: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True); S = pd.read_csv(res / "summary.csv")
    # 1 categoriser robustness
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    bars(ax[0], S, [("time_majority", "always\nmajority"), ("time_lookup", "exact\nlookup"), ("time_model", "n-gram\nmodel")], [GREY, GREY, TEAL], ylim=(0, 105), title="Later months, clean strings")
    bars(ax[1], S, [("messy_lookup", "exact\nlookup"), ("messy_nn", "fuzzy\nnearest"), ("messy_model", "n-gram\nmodel")], [GREY, AMBER, TEAL], ylim=(0, 105), title="Same, with typos")
    bars(ax[2], S, [("heldout_majority", "always\nmajority"), ("heldout_nn", "fuzzy\nnearest"), ("heldout_model", "n-gram\nmodel")], [GREY, AMBER, TEAL], ylim=(0, 105), title="Merchant never seen")
    ax[0].set_ylabel("accuracy, %"); fig.suptitle("Categoriser accuracy: mean of 10 simulated users, 95% interval", x=.01, ha="left", fontsize=11, color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out / "categoriser.png"); plt.close(fig)
    # 2 risk-coverage
    cov = [100, 90, 80, 70, 60, 50]; v = [_g(S, f"rc_{c}") for c in cov]
    fig, ax = plt.subplots(figsize=(6.2, 3.9)); ax.plot(cov, [a[0] * 100 for a in v], "-o", color=TEAL)
    ax.fill_between(cov, [a[1] * 100 for a in v], [a[2] * 100 for a in v], color=TEAL, alpha=.18)
    for c, a in zip(cov, v): ax.text(c, a[0] * 100 - 2.4, f"{a[0]*100:.0f}", ha="center", fontsize=9)
    ax.invert_xaxis(); ax.set_ylim(80, 101); ax.set_xlabel("share of transactions auto-categorised (most confident first), %"); ax.set_ylabel("accuracy on those, %")
    ax.set_title("Send the least confident to review, and accuracy climbs", loc="left"); fig.tight_layout(); fig.savefig(out / "risk_coverage.png"); plt.close(fig)
    # 3 active learning
    ns = [60, 120, 180, 240, 300, 360, 420]; fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for k, c, lab in (("random", GREY, "random rows"), ("uncertainty", TEAL, "least-confident rows")):
        v = [_g(S, f"al_{k}_{n}") for n in ns]; ax.plot(ns, [a[0] * 100 for a in v], "-o", color=c, label=lab)
        ax.fill_between(ns, [a[1] * 100 for a in v], [a[2] * 100 for a in v], color=c, alpha=.15)
    ax.set_xlabel("labelled transactions"); ax.set_ylabel("accuracy on later months, %"); ax.legend(frameon=False, loc="lower right")
    ax.set_title("Which rows to label first? Random wins here", loc="left"); fig.tight_layout(); fig.savefig(out / "active_learning.png"); plt.close(fig)
    # 4 forecast
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); names = [("naive", "naive"), ("seasonal_naive", "seasonal\nnaive"), ("ma3", "3-month\naverage"), ("ses", "exp.\nsmoothing"), ("rf", "random\nforest"), ("ensemble", "ensemble")]
    bars(ax, S, [(f"wape_{k}", l) for k, l in names], [GREY, GREY, GREY, AMBER, AMBER, TEAL], pct=False, ylim=(0, 29), title="Next-month spend: error (WAPE, lower is better)")
    fig.tight_layout(); fig.savefig(out / "forecast.png"); plt.close(fig)
    # 5 anomalies
    A = pd.read_csv(res / "anomaly_sweep.csv"); fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for m, c, lab in (("per_category", TEAL, "per-category z-score"), ("global", GREY, "one global z-score")):
        a = A[A.method == m]; ax.plot(a.z, a.f1 * 100, "-o", color=c, label=lab)
    ax.set_xlabel("flag if robust z-score above"); ax.set_ylabel("F1 on injected anomalies, %"); ax.set_ylim(0, 100); ax.legend(frameon=False)
    ax.set_title("A normal for each category beats one normal for all", loc="left"); fig.tight_layout(); fig.savefig(out / "anomaly.png"); plt.close(fig)


def forecast_fan(T_total: pd.Series, nm: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(T_total)); ax.plot(x, T_total.values, color=NAVY, lw=1.8)
    ax.errorbar([len(T_total)], [nm["point"]], yerr=[[nm["point"] - nm["lo"]], [nm["hi"] - nm["point"]]], fmt="o", color=AMBER, capsize=5, lw=2)
    ax.set_xticks(list(x[::6]) + [len(T_total)]); ax.set_xticklabels(list(T_total.index[::6]) + [nm["month"]], rotation=0, fontsize=8)
    ax.set_title("Monthly spend and next-month forecast (80% band)", loc="left"); fig.tight_layout(); fig.savefig(path); plt.close(fig)
