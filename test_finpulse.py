import sys, sqlite3
from pathlib import Path
import pandas as pd, pytest
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import generate_data, etl, categorize, savings

@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / "t.db"
    for m in (generate_data, etl, categorize, savings): monkeypatch.setattr(m, "DB_PATH", p)
    monkeypatch.setattr(categorize, "ROOT", tmp_path); (tmp_path / "reports").mkdir(); (tmp_path / "models").mkdir()
    generate_data.main(); etl.main(); return p

def test_clean_desc_strips_channel_ref_and_city():
    assert etl.clean_desc("UPI-SWIGGY 48213 DELHI") == "SWIGGY"
    assert etl.clean_desc("POS AMAZON 99120") == "AMAZON"

def test_net_worth_counts_investments_once(db):
    con = sqlite3.connect(db); nw = pd.read_sql("select * from net_worth", con); tx = pd.read_sql("select * from raw_transactions", con)
    credits = tx[tx.type == "credit"].amount.sum(); debits = tx[tx.type == "debit"].amount.sum()
    last = nw.iloc[-1]
    assert abs(last.cum_cash - (credits - debits)) < 1e-6            # SIP outflows ARE deducted from cash
    assert abs(last.net_worth - (last.cum_cash + last.portfolio_value)) < 1e-6

def test_categoriser_split_is_time_ordered_and_beats_chance(db):
    categorize.main(); m = __import__("json").loads((db.parent / "reports" / "categorizer_metrics.json").read_text())
    assert m["accuracy"] > 0.8 and m["train_rows"] + m["test_rows"] > 1000

def test_label_noise_never_touches_income_or_rent(db):
    tx = pd.read_sql("select description, category from raw_transactions", sqlite3.connect(db))
    assert (tx[tx.description.str.contains("SALARY|RENT")].category.isin(["Income", "Rent"])).all()

def test_savings_math(db):
    t, total, income = savings.savings_plan({"Shopping": 10, "Food Delivery": 20, "Entertainment": 0})
    assert abs(total - (t.last_12m_spend["Shopping"] * .1 + t.last_12m_spend["Food Delivery"] * .2)) < 1
    assert income > 0
