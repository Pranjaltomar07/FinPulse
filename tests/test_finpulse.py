import dataclasses, sqlite3
import numpy as np, pandas as pd, pytest
from finpulse.config import SimConfig, SPEND_CATS
from finpulse.simulate import simulate
from finpulse.clean import clean_desc, prepare
from finpulse.ingest import load_statement
from finpulse.rules import seed_label
from finpulse import categorize as C, recurring as R, anomaly as A, forecast as F, budget as B, warehouse as W, pipeline as P, dashboard as D


@pytest.fixture(scope="module")
def sim(): return simulate()
@pytest.fixture(scope="module")
def df(sim): return prepare(sim.raw)


# ---- simulator
def test_deterministic(): assert simulate().raw.equals(simulate().raw)
def test_seed_changes_data(): assert not simulate(SimConfig(seed=1)).raw.amount.equals(simulate(SimConfig(seed=2)).raw.amount)
def test_truth_separate_from_raw(sim): assert not {"is_recurring", "is_anomaly", "merchant"} & set(sim.raw.columns)
def test_truth_aligned(sim): assert (sim.raw.txn_id == sim.truth.txn_id).all()
def test_anomaly_count(sim): assert sim.truth.is_anomaly.sum() == 10
def test_late_merchants_exist(df):
    cut = sorted(df.month.unique())[27]; a, b = df[df.month < cut], df[df.month >= cut]
    assert (~b.clean_desc.isin(set(a.clean_desc))).mean() > .1
def test_dates_sorted(sim): assert sim.raw.date.is_monotonic_increasing

# ---- cleaning
def test_clean_strips_prefix_and_refs(): assert clean_desc("UPI-SWIGGY 48213 DELHI") == "SWIGGY"
def test_clean_pos(): assert clean_desc("POS DMART 12345 NOIDA") == "DMART"
def test_clean_idempotent(): assert clean_desc(clean_desc("NEFT-NETFLIX 99999")) == "NETFLIX"
def test_prepare_adds_columns(df): assert {"clean_desc", "month"} <= set(df.columns)

# ---- ingest and rules
def test_ingest_debit_credit(tmp_path):
    f = tmp_path / "s.csv"; f.write_text("Date,Narration,Withdrawal,Deposit\n05/01/2024,UPI-SWIGGY 12345,250.5,\n01/02/2024,SALARY ACME,,50000\n")
    d = load_statement(f, "Date", "Narration", debit_col="Withdrawal", credit_col="Deposit")
    assert list(d.type) == ["debit", "credit"] and d.amount.tolist() == [250.5, 50000.0] and d.date.iloc[0] == "2024-01-05"
def test_ingest_signed(tmp_path):
    f = tmp_path / "s.csv"; f.write_text("d,t,a\n2024-01-05,COFFEE,-120\n2024-01-06,REFUND,30\n")
    d = load_statement(f, "d", "t", amount_col="a"); assert list(d.type) == ["debit", "credit"] and (d.amount > 0).all()
def test_rules_known(): assert seed_label("SWIGGY", "debit") == "Food Delivery"
def test_rules_unknown(): assert seed_label("QWERTY ZXCV", "debit") is None
def test_rules_credit_income(): assert seed_label("QWERTY ZXCV", "credit") == "Income"

# ---- categoriser
def _split(df):
    cut = sorted(df.month.unique())[27]; return df[df.month < cut], df[df.month >= cut]
def test_model_beats_majority(df):
    a, b = _split(df); p, _ = C.Categoriser().fit(a.clean_desc, a.category).predict(b.clean_desc)
    assert C.accuracy(b.category, p) > C.accuracy(b.category, np.repeat(a.category.mode()[0], len(b))) + .3
def test_confidence_in_unit_interval(df):
    a, b = _split(df); _, c = C.Categoriser().fit(a.clean_desc, a.category).predict(b.clean_desc); assert ((c > 0) & (c <= 1)).all()
def test_risk_coverage_monotone_up(df):
    a, b = _split(df); p, c = C.Categoriser().fit(a.clean_desc, a.category).predict(b.clean_desc)
    rc = C.risk_coverage(b.category, p, c); assert rc[.5] >= rc[1.0]
def test_learn_adds_rows(df):
    a, _ = _split(df); m = C.Categoriser().fit(a.clean_desc[:50], a.category[:50]); m2 = m.learn(["TOTALLY NEW SHOP"], ["Shopping"])
    assert len(m2.texts) == 51 and len(m.texts) == 50
def test_corrupt_changes_string():
    rng = np.random.default_rng(0); assert all(C.corrupt("BIGBASKET", rng) != "BIGBASKET" for _ in range(30))
def test_lookup_unseen_falls_back_to_majority():
    assert C.lookup_predict(["A", "A", "B"], ["x", "x", "y"], ["ZZZ"])[0] == "x"

# ---- recurring
def test_recurring_perfect_on_sim(sim, df):
    s = A.prf(sim.truth.is_recurring, R.flag_rows(df, R.detect_recurring(df))); assert s["precision"] == 1 and s["recall"] == 1
def test_variable_bill_not_recurring():
    d = pd.DataFrame({"clean_desc": ["ELEC"] * 8, "date": pd.date_range("2024-01-05", periods=8, freq="30D").astype(str),
                      "amount": [400, 900, 300, 1200, 500, 950, 350, 1100], "type": "debit"}); assert R.detect_recurring(d).empty
def test_weekly_not_recurring():
    d = pd.DataFrame({"clean_desc": ["GYM"] * 10, "date": pd.date_range("2024-01-01", periods=10, freq="7D").astype(str), "amount": 500.0, "type": "debit"}); assert R.detect_recurring(d).empty

# ---- anomaly
def test_anomaly_finds_most(sim, df):
    s = A.prf(sim.truth.is_anomaly, A.score(df) > 4); assert s["recall"] >= .5 and s["precision"] >= .5
def test_anomaly_beats_global(sim, df):
    assert A.prf(sim.truth.is_anomaly, A.score(df) > 4)["f1"] > A.prf(sim.truth.is_anomaly, A.score(df, False) > 4)["f1"]
def test_fixed_categories_not_scored(df): assert A.score(df)[df.category.isin(["Rent", "Income"])].isna().all()

# ---- forecast
def test_wape_zero_when_perfect(): assert F.wape([1, 2, 3], [1, 2, 3]) == 0
def test_wape_known(): assert F.wape([100], [90]) == pytest.approx(10)
def test_ses_constant(): assert F.ses(np.array([5.0] * 6), .3) == pytest.approx(5)
def test_features_use_past_only():
    s = pd.Series(np.arange(24.0), index=[f"2023-{m:02d}" if m <= 12 else f"2024-{m-12:02d}" for m in range(1, 25)]); d = F.feats(s)
    assert (d.lag1 == d.y - 1).all()
def test_backtest_shapes(df):
    T = F.monthly_totals(df); act, Pr = F.rolling_origin(T.TOTAL, 0); assert all(len(v) == len(act) for v in Pr.values()) and len(act) >= 10
def test_ensemble_beats_naive(df):
    T = F.monthly_totals(df); act, Pr = F.rolling_origin(T.TOTAL, 0); assert F.wape(act, Pr["ensemble"]) < F.wape(act, Pr["naive"])
def test_interval_ordered(df):
    nm = F.next_month(F.monthly_totals(df).TOTAL); assert nm["lo"] <= nm["point"] <= nm["hi"]

# ---- budget
def test_budget_risk_bounds():
    assert B.budget_risk(100, [-5, 0, 5], 1000) == 0 and B.budget_risk(100, [-5, 0, 5], 0) == 1
def test_budget_risk_empty_is_nan(): assert np.isnan(B.budget_risk(1, [], 2))
def test_savings_plan_arithmetic(df):
    out, tot, inc = B.savings_plan(df, {"Shopping": 10}); assert tot == pytest.approx(out.loc["Shopping", "last_12m_spend"] * .1, abs=1)

# ---- warehouse, pipeline, dashboard
def test_warehouse_views(df, tmp_path):
    db = W.build(df, tmp_path / "t.db"); v = W.query(db, "select * from v_monthly_summary")
    assert len(v) == df.month.nunique() and set(["income", "outflow", "net"]) <= set(v.columns)
def test_sql_matches_pandas(df, tmp_path):
    db = W.build(df, tmp_path / "t.db"); s = W.query(db, "select sum(outflow) o from v_monthly_summary").o[0]
    assert s == pytest.approx(df[df.type == "debit"].amount.sum(), rel=1e-3)
def test_pipeline_and_dashboard(sim, tmp_path):
    res = P.run(sim.raw, tmp_path); D.render(res, tmp_path / "d.html"); html = (tmp_path / "d.html").read_text()
    assert "FinPulse" in html and "Review queue" in html and res["db"].exists()
def test_cold_start_pipeline(sim, tmp_path):
    raw = sim.raw.drop(columns="category"); res = P.run(raw, tmp_path, cold=True)
    assert res["df"].category.notna().all() and (res["df"].category == "Income").any()
def test_review_flags_low_confidence(sim, tmp_path):
    df = P.run(sim.raw, tmp_path)["df"]; assert (df[df.needs_review].confidence < P.CONF_REVIEW).all()
