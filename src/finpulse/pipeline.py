"""End-to-end run on one user's statement: clean, categorise, detect, forecast, store, report."""
from pathlib import Path
import numpy as np, pandas as pd
from .clean import prepare
from .rules import seed_label
from . import categorize as C, recurring as R, anomaly as A, forecast as F, budget as B, warehouse as W
from .config import DISCRETIONARY

CONF_REVIEW = 0.6


def categorise(df: pd.DataFrame, labelled: pd.Series) -> pd.DataFrame:
    """Fill in categories. `labelled` marks rows whose category is already known (user-confirmed or from the
    statement); the rest get a model prediction with a confidence, and low-confidence rows are queued for review."""
    df = df.copy(); df["confidence"] = 1.0; df["source"] = np.where(labelled, "given", "model")
    if (~labelled).any():
        m = C.Categoriser().fit(df.loc[labelled, "clean_desc"], df.loc[labelled, "category"])
        p, c = m.predict(df.loc[~labelled, "clean_desc"]); df.loc[~labelled, "category"] = p; df.loc[~labelled, "confidence"] = c
    df["needs_review"] = df.confidence < CONF_REVIEW; return df


def cold_start(df: pd.DataFrame) -> pd.DataFrame:
    """Statement with no categories at all: seed labels from keyword rules, then let the model learn from them."""
    df = df.copy(); df["category"] = [seed_label(d, t) for d, t in zip(df.clean_desc, df.type)]
    known = df.category.notna(); df.loc[~known, "category"] = None
    out = categorise(df.assign(category=df.category.fillna("")), known); return out


def run(raw: pd.DataFrame, out: Path, labelled_share_cut: float | None = .75, cold: bool = False) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = prepare(raw)
    if cold or "category" not in df.columns:
        df = cold_start(df)
    else:
        cut = sorted(df.month.unique())[int(len(df.month.unique()) * labelled_share_cut)] if labelled_share_cut else "9999"
        df = categorise(df, df.month < cut)
    found = R.detect_recurring(df)
    df["z"] = A.score(df); anom = df[df.z > 4.0][["txn_id", "date", "description", "amount", "category", "z"]].round(2)
    T = F.monthly_totals(df); nm = F.next_month(T.TOTAL) if len(T) >= 12 else None
    Wdb = W.build(df.drop(columns=["z"]), out / "finpulse.db", found, anom)
    return {"df": df, "recurring": found, "anomalies": anom, "monthly": T, "forecast": nm, "db": Wdb}
