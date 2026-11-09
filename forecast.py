"""Forecast monthly spend per category (Random Forest on lag + seasonality features) and
compare against naive baselines on the last 8 months. An LSTM is a drop-in alternative
(see README) but with only 36 monthly points a tree model is the sounder choice."""
import sqlite3, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import *

def feats(s: pd.Series):
    d = pd.DataFrame({"y": s}); d["month"] = pd.to_datetime(d.index + "-01").month
    for l in (1, 2, 3): d[f"lag{l}"] = d.y.shift(l)
    d["roll3"] = d.y.shift(1).rolling(3).mean()
    d["sin"] = np.sin(2 * np.pi * d.month / 12); d["cos"] = np.cos(2 * np.pi * d.month / 12)
    return d.dropna()

def wape(a, p):  # weighted abs % error: stable when some months are ~0 (MAPE explodes there)
    a, p = np.asarray(a), np.asarray(p); return float(np.abs(a - p).sum() / max(a.sum(), 1) * 100)

def main(test_n=8):
    con = sqlite3.connect(DB_PATH)
    ms = pd.read_sql("select * from monthly_summary where category not in ('Income','Investment')", con)
    piv = ms.pivot(index="month", columns="category", values="total").fillna(0)
    piv["TOTAL"] = piv.sum(axis=1)
    res, nxt = [], {}
    for cat in piv.columns:
        d = feats(piv[cat]); X = d.drop(columns="y"); y = d.y
        tr, te = slice(0, len(d) - test_n), slice(len(d) - test_n, None)
        rf = RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED).fit(X[tr], y[tr])
        res.append({"category": cat, "rf_wape": wape(y[te], rf.predict(X[te])),
                    "naive_last_month_wape": wape(y[te], X[te].lag1), "moving_avg3_wape": wape(y[te], X[te].roll3)})
        rf.fit(X, y)  # refit on everything, predict next month
        last = piv[cat]; nm = (pd.Period(piv.index[-1]) + 1)
        row = pd.DataFrame([{"month": nm.month, "lag1": last.iloc[-1], "lag2": last.iloc[-2], "lag3": last.iloc[-3],
                             "roll3": last.iloc[-3:].mean(), "sin": np.sin(2*np.pi*nm.month/12), "cos": np.cos(2*np.pi*nm.month/12)}])[X.columns]
        nxt[cat] = round(float(rf.predict(row)[0]), 2)
    r = pd.DataFrame(res).round(1); print(r.to_string(index=False)); print("next month:", nxt)
    r.to_csv(ROOT / "reports" / "forecast_eval.csv", index=False)
    pd.Series(nxt, name="forecast").to_csv(ROOT / "reports" / "next_month_forecast.csv")

if __name__ == "__main__":
    main()
