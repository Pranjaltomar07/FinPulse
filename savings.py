"""Savings simulation on this user's own history: what if discretionary categories were
trimmed by X%? Uses the last 12 months of actual spend. Output is a what-if, not a promise."""
import sqlite3, pandas as pd
from config import *

def savings_plan(cuts: dict, months=12):
    ms = pd.read_sql("select * from monthly_summary", sqlite3.connect(DB_PATH))
    last = ms[ms.month >= sorted(ms.month.unique())[-months]]
    spend = last.groupby("category").total.sum()
    out = pd.DataFrame({"last_12m_spend": spend.reindex(DISCRETIONARY).round(0)})
    out["cut_pct"] = [cuts.get(c, 0) for c in out.index]
    out["annual_saving"] = (out.last_12m_spend * out.cut_pct / 100).round(0)
    income = last[last.category == "Income"].total.sum()
    return out, float(out.annual_saving.sum()), float(income)

if __name__ == "__main__":
    for cuts in ({c: 10 for c in DISCRETIONARY}, {c: 20 for c in DISCRETIONARY}):
        t, s, inc = savings_plan(cuts); print(t, f"\nTotal annual saving: Rs {s:,.0f} ({100*s/inc:.1f}% of income)\n")
