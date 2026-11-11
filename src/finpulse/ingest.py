"""Load a real bank-statement CSV into the standard frame used everywhere else.

Standard columns: date (YYYY-MM-DD), description, amount (always positive), type ('debit' or 'credit'),
and optionally category. Statements come in two common shapes: one signed amount column (negative =
money out) or separate debit and credit columns. Pass whichever you have.
"""
import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[,\s₹]|Rs\.?|INR", "", regex=True).replace({"": None, "nan": None}),
                         errors="coerce")


def load_statement(source, date_col, desc_col, amount_col=None, debit_col=None, credit_col=None,
                   category_col=None, dayfirst=True) -> pd.DataFrame:
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    need = [c for c in (date_col, desc_col, amount_col, debit_col, credit_col) if c]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) not found in the file: {missing}. Available: {list(df.columns)}")
    if amount_col:
        signed = _num(df[amount_col])
        amount, typ = signed.abs(), signed.map(lambda v: "debit" if v < 0 else "credit")
    elif debit_col and credit_col:
        deb, cre = _num(df[debit_col]).fillna(0), _num(df[credit_col]).fillna(0)
        amount = deb.where(deb > 0, cre); typ = (deb > 0).map({True: "debit", False: "credit"})
    else:
        raise ValueError("give either amount_col (signed) or both debit_col and credit_col")
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce").dt.date.astype("string"),
                        "description": df[desc_col].astype(str), "amount": amount.round(2), "type": typ})
    if category_col:
        out["category"] = df[category_col]
    out = out.dropna(subset=["date", "amount"]).query("amount > 0").sort_values("date", kind="stable").reset_index(drop=True)
    out["date"] = out.date.astype(str)
    out.insert(0, "txn_id", range(1, len(out) + 1))
    return out
