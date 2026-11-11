"""SQLite warehouse: transactions in, SQL views out (the same numbers the dashboard shows)."""
import sqlite3
from pathlib import Path
import pandas as pd

VIEWS = {
 "v_monthly_category": """CREATE VIEW v_monthly_category AS SELECT month, category, ROUND(SUM(amount),2) AS spend, COUNT(*) AS n
   FROM transactions WHERE type='debit' AND category NOT IN ('Rent','Investment') GROUP BY month, category""",
 "v_monthly_summary": """CREATE VIEW v_monthly_summary AS SELECT month,
   ROUND(SUM(CASE WHEN type='credit' THEN amount END),2) AS income,
   ROUND(SUM(CASE WHEN type='debit' THEN amount END),2) AS outflow,
   ROUND(SUM(CASE WHEN type='credit' THEN amount ELSE -amount END),2) AS net FROM transactions GROUP BY month""",
 "v_top_merchants": """CREATE VIEW v_top_merchants AS SELECT clean_desc AS merchant, category, COUNT(*) AS n, ROUND(SUM(amount),2) AS total
   FROM transactions WHERE type='debit' GROUP BY clean_desc, category ORDER BY total DESC""",
}


def build(df: pd.DataFrame, path, recurring: pd.DataFrame | None = None, anomalies: pd.DataFrame | None = None) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    df.to_sql("transactions", con, index=False)
    if recurring is not None: recurring.to_sql("recurring", con, index=False)
    if anomalies is not None: anomalies.to_sql("anomalies", con, index=False)
    for sql in VIEWS.values(): con.execute(sql)
    con.commit(); con.close(); return path


def query(path, sql: str) -> pd.DataFrame:
    con = sqlite3.connect(path)
    try: return pd.read_sql(sql, con)
    finally: con.close()
