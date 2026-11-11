"""Flag unusually large purchases with a robust z-score on log amounts."""
import numpy as np, pandas as pd
from .config import FIXED


def _rz(x: np.ndarray) -> np.ndarray:
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 0.15)          # floor stops tiny spreads from exploding the score


def score(df: pd.DataFrame, by_category: bool = True) -> pd.Series:
    """Robust z-score of log(amount) for ordinary debits; NaN for rent, income and investments."""
    z = pd.Series(np.nan, index=df.index); ok = (df.type == "debit") & ~df.category.isin(FIXED)
    logs = np.log(df.loc[ok, "amount"].to_numpy(float))
    if not by_category:
        z[ok] = _rz(logs); return z
    for _, idx in df[ok].groupby("category").groups.items():
        z[idx] = _rz(np.log(df.loc[idx, "amount"].to_numpy(float)))
    return z


def prf(truth: pd.Series, flagged: pd.Series) -> dict:
    t, f = truth.to_numpy(bool), flagged.to_numpy(bool); tp = int((t & f).sum())
    p = tp / f.sum() if f.sum() else 0.0; r = tp / t.sum() if t.sum() else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
