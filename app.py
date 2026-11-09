import sqlite3, json, sys
from pathlib import Path
import pandas as pd, plotly.express as px, streamlit as st
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import DB_PATH, ROOT, DISCRETIONARY
from savings import savings_plan

st.set_page_config("FinPulse", layout="wide")
con = sqlite3.connect(DB_PATH)
tx = pd.read_sql("select * from clean_transactions", con)
nw = pd.read_sql("select * from net_worth", con)
fc = pd.read_csv(ROOT / "reports" / "forecast_eval.csv"); nxt = pd.read_csv(ROOT / "reports" / "next_month_forecast.csv", index_col=0)
cm = json.loads((ROOT / "reports" / "categorizer_metrics.json").read_text())

st.title("FinPulse - personal finance intelligence")
st.caption("Synthetic transactions. Categoriser and forecast metrics are on a time-based holdout.")
t1, t2, t3, t4 = st.tabs(["Spending", "Net worth", "Forecast", "Savings what-if"])
spend = tx[~tx.category.isin(["Income", "Investment"])]
with t1:
    m = spend.groupby(["month", "category"]).amount.sum().reset_index()
    st.plotly_chart(px.bar(m, x="month", y="amount", color="category", title="Monthly spend by category"), width="stretch")
    c = st.columns(3); c[0].metric("Categoriser accuracy", f"{cm['accuracy']:.1%}"); c[1].metric("Macro F1", cm["macro_f1"]); c[2].metric("Transactions", len(tx))
    mis = tx[tx.category != tx.predicted_category][["date", "description", "category", "predicted_category"]]
    with st.expander(f"Misclassified rows ({len(mis)})"): st.dataframe(mis)
with t2:
    st.plotly_chart(px.line(nw, x="month", y=["cum_cash", "portfolio_value", "net_worth"], title="Cash + investments"), width="stretch")
with t3:
    st.metric("Forecast total spend next month", f"Rs {nxt.loc['TOTAL','forecast']:,.0f}")
    st.dataframe(fc.round(1), width="stretch"); st.caption("WAPE = weighted absolute % error on the last 8 months. Lower is better.")
with t4:
    cols = st.columns(3); cuts = {c: cols[i].slider(f"Cut {c} %", 0, 50, 10) for i, c in enumerate(DISCRETIONARY)}
    table, total, income = savings_plan(cuts)
    st.dataframe(table); st.metric("Projected annual saving", f"Rs {total:,.0f}", f"{100*total/income:.1f}% of income")
    st.warning("What-if based on your last 12 months. It assumes the cuts are achievable and spend doesn't rebound.")
