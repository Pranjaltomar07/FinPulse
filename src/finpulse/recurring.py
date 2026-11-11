"""Detect subscriptions and other monthly recurring payments from transaction history."""
import numpy as np, pandas as pd


def detect_recurring(df: pd.DataFrame, min_occ=4, interval=(24, 38), max_gap_mad=5.0, max_amount_cv=0.12) -> pd.DataFrame:
    """A merchant is recurring if it repeats about monthly (median gap in `interval` days, small spread)
    for a near-constant amount. Variable bills such as electricity fail the amount test by design."""
    d = df.assign(dt=pd.to_datetime(df.date)).sort_values("dt"); rows = []
    for key, g in d.groupby("clean_desc"):
        if len(g) < min_occ: continue
        gaps = g.dt.diff().dt.days.dropna().to_numpy(float); med = float(np.median(gaps))
        mad = float(np.median(np.abs(gaps - med))); amt = g.amount.to_numpy(float); cv = float(amt.std() / amt.mean())
        if interval[0] <= med <= interval[1] and mad <= max_gap_mad and cv <= max_amount_cv:
            rows.append({"merchant": key, "kind": "income" if g.type.iloc[-1] == "credit" else "expense", "n": len(g),
                         "median_gap_days": round(med, 1), "median_amount": round(float(np.median(amt)), 2),
                         "amount_cv": round(cv, 3), "monthly_cost": round(float(np.median(amt)) * 30.4 / med, 2),
                         "last_seen": g.dt.iloc[-1].date().isoformat(),
                         "next_expected": (g.dt.iloc[-1] + pd.Timedelta(days=round(med))).date().isoformat()})
    return pd.DataFrame(rows, columns=["merchant", "kind", "n", "median_gap_days", "median_amount", "amount_cv",
                                       "monthly_cost", "last_seen", "next_expected"]).sort_values("monthly_cost", ascending=False, ignore_index=True)


def flag_rows(df: pd.DataFrame, found: pd.DataFrame) -> pd.Series:
    return df.clean_desc.isin(set(found.merchant))
