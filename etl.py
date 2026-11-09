"""Clean raw bank rows -> analytics layer (monthly summary, net worth).
Plain pandas + SQL here; the same steps drop into an Airflow DAG."""
import re, sqlite3, pandas as pd
from config import *

def clean_desc(s: str) -> str:
    s = s.upper()
    s = re.sub(r"\b(UPI|POS|NEFT|IMPS|ECOM)\b[-/ ]?", " ", s)   # channel prefixes
    s = re.sub(r"\d{4,}", " ", s)                                # reference numbers
    s = re.sub(r"\b(DELHI|GURGAON|NOIDA|BLR|MUMBAI)\b", " ", s)  # city codes
    return re.sub(r"[^A-Z ]", " ", s).split().__str__().replace("'", "").strip("[]").replace(",", "")

def main():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("select * from raw_transactions", con)
    df["clean_desc"] = df.description.map(clean_desc)
    df["month"] = df.date.str[:7]
    df.to_sql("clean_transactions", con, if_exists="replace", index=False)
    con.executescript("""
    DROP TABLE IF EXISTS monthly_summary;
    CREATE TABLE monthly_summary AS
    SELECT month, category, SUM(amount) AS total, COUNT(*) AS n FROM clean_transactions GROUP BY 1,2;
    DROP TABLE IF EXISTS net_worth;
    CREATE TABLE net_worth AS
    WITH cf AS (SELECT month, SUM(CASE WHEN type='credit' THEN amount ELSE -amount END) AS net_cash
                FROM clean_transactions GROUP BY 1)
    SELECT cf.month, cf.net_cash,
           SUM(cf.net_cash) OVER (ORDER BY cf.month) AS cum_cash,
           i.portfolio_value,
           SUM(cf.net_cash) OVER (ORDER BY cf.month) + COALESCE(i.portfolio_value,0) AS net_worth
    FROM cf LEFT JOIN raw_investments i ON substr(i.month,1,7)=cf.month;""")
    con.commit(); print("clean_transactions:", len(df), "| monthly_summary & net_worth built")

if __name__ == "__main__":
    main()
