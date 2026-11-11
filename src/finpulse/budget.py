"""What-if savings and budget risk."""
import numpy as np, pandas as pd
from .config import DISCRETIONARY


def savings_plan(df: pd.DataFrame, cuts: dict, months: int = 12):
    """Annual saving if discretionary categories were cut by the given percentages, from the last `months` of spend."""
    last = sorted(df.month.unique())[-months:]; d = df[df.month.isin(last)]
    spend = d[(d.type == "debit") & d.category.isin(DISCRETIONARY)].groupby("category").amount.sum().reindex(DISCRETIONARY).fillna(0)
    out = pd.DataFrame({"last_12m_spend": spend.round(0)}); out["cut_pct"] = [cuts.get(c, 0) for c in out.index]
    out["annual_saving"] = (out.last_12m_spend * out.cut_pct / 100).round(0)
    income = float(d[d.type == "credit"].amount.sum()); return out, float(out.annual_saving.sum()), income


def budget_risk(point: float, residuals, budget: float) -> float:
    """Chance next month's spend exceeds `budget`, using the forecast's own backtest errors (actual - predicted)."""
    r = np.asarray(residuals, float)
    return float("nan") if len(r) == 0 else float((point + r > budget).mean())
