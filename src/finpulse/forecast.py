"""Monthly spend forecasting: simple baselines, a random forest, an ensemble, and honest intervals."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .config import SPEND_CATS

MODELS = ("naive", "seasonal_naive", "ma3", "ses", "rf", "ensemble")


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    sp = df[(df.type == "debit") & df.category.isin(SPEND_CATS)]
    piv = sp.pivot_table(index="month", columns="category", values="amount", aggfunc="sum").fillna(0.0)
    piv["TOTAL"] = piv.sum(axis=1); return piv


def feats(s: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"y": s}); d["month"] = pd.to_datetime(d.index + "-01").month
    for l in (1, 2, 3): d[f"lag{l}"] = d.y.shift(l)
    d["roll3"] = d.y.shift(1).rolling(3).mean()
    d["sin"] = np.sin(2 * np.pi * d.month / 12); d["cos"] = np.cos(2 * np.pi * d.month / 12)
    return d.dropna()


def wape(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float); return float(np.abs(a - p).sum() / max(a.sum(), 1) * 100)


def ses(history, alpha):
    level = history[0]
    for y in history[1:]: level = alpha * y + (1 - alpha) * level
    return level


def best_alpha(history) -> float:
    def err(a):
        lvl, e = history[0], 0.0
        for y in history[1:]: e += abs(y - lvl); lvl = a * y + (1 - a) * lvl
        return e
    grid = np.linspace(.05, .95, 19); return float(grid[np.argmin([err(a) for a in grid])])


def one_step(series: np.ndarray, X: pd.DataFrame, y: np.ndarray, i: int, t: int, seed: int, n_trees=100) -> dict:
    """Predict month t (row i of the feature table) from information available before it."""
    hist = series[:t]
    rf = RandomForestRegressor(n_trees, min_samples_leaf=2, random_state=seed).fit(X.iloc[:i], y[:i])
    p = {"naive": hist[-1], "seasonal_naive": hist[-12], "ma3": hist[-3:].mean(),
         "ses": ses(hist, best_alpha(hist)), "rf": float(rf.predict(X.iloc[[i]])[0])}
    p["ensemble"] = float(np.mean([p["ma3"], p["ses"], p["rf"]])); return p


def rolling_origin(s: pd.Series, seed: int, first_origin: int = 18) -> tuple[np.ndarray, dict]:
    """Backtest many forecast origins, each using only the past. Returns actuals and predictions per model."""
    d = feats(s); X, y = d.drop(columns="y"), d.y.to_numpy(); series = s.to_numpy(float); pos = {m: i for i, m in enumerate(s.index)}
    P = {k: [] for k in MODELS}; A = []
    for i in range(first_origin, len(d)):
        t = pos[d.index[i]]; pr = one_step(series, X, y, i, t, seed)
        for k in MODELS: P[k].append(pr[k])
        A.append(series[t])
    return np.array(A), {k: np.array(v) for k, v in P.items()}


def interval_backtest(actual, pred, level=.8, warmup=6) -> dict:
    """Split-conformal style: the band at each step uses only earlier absolute errors."""
    res = np.abs(actual - pred); hit, width = [], []
    for j in range(warmup, len(actual)):
        q = np.quantile(res[:j], level, method="higher"); hit.append(res[j] <= q); width.append(2 * q)
    return {"coverage": float(np.mean(hit)), "width_pct": float(np.mean(width) / actual.mean() * 100)}


def next_month(s: pd.Series, seed: int = 0, level: float = .8) -> dict:
    """Ensemble forecast for the month after the history, with an interval from backtest errors."""
    d = feats(s); X, y = d.drop(columns="y"), d.y.to_numpy(); series = s.to_numpy(float)
    nm = pd.Period(s.index[-1]) + 1
    row = pd.DataFrame([{"month": nm.month, "lag1": series[-1], "lag2": series[-2], "lag3": series[-3],
                         "roll3": series[-3:].mean(), "sin": np.sin(2 * np.pi * nm.month / 12),
                         "cos": np.cos(2 * np.pi * nm.month / 12)}])[X.columns]
    rf = RandomForestRegressor(100, min_samples_leaf=2, random_state=seed).fit(X, y)
    parts = [series[-3:].mean(), ses(series, best_alpha(series)), float(rf.predict(row)[0])]
    point = float(np.mean(parts)); resid = np.array([])
    if len(d) > 22:
        A, P = rolling_origin(s, seed); resid = A - P["ensemble"]
    q = float(np.quantile(np.abs(resid), level, method="higher")) if len(resid) else np.nan
    return {"month": str(nm), "point": point, "lo": max(0.0, point - q), "hi": point + q, "residuals": resid}
