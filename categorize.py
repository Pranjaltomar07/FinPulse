"""Expense categoriser: char n-gram TF-IDF + logistic regression on cleaned descriptions.
Evaluated on a time-based split (train on earlier months, test on later) so it can't
memorise future rows."""
import sqlite3, json, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import *

def main():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("select * from clean_transactions", con)
    cut = sorted(df.month.unique())[int(N_MONTHS * .75)]
    tr, te = df[df.month < cut], df[df.month >= cut]
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=10))
    pipe.fit(tr.clean_desc, tr.category); pred = pipe.predict(te.clean_desc)
    # baseline: most common category per exact cleaned string (lookup table)
    lut = tr.groupby("clean_desc").category.agg(lambda s: s.mode()[0]); 
    base = te.clean_desc.map(lut).fillna("Shopping")
    m = {"train_rows": len(tr), "test_rows": len(te),
         "accuracy": round(accuracy_score(te.category, pred), 4),
         "macro_f1": round(f1_score(te.category, pred, average="macro"), 4),
         "lookup_baseline_accuracy": round(accuracy_score(te.category, base), 4)}
    print(json.dumps(m, indent=2)); print(classification_report(te.category, pred, zero_division=0))
    (ROOT / "reports" / "categorizer_metrics.json").write_text(json.dumps(m, indent=2))
    (ROOT / "models").mkdir(exist_ok=True); pipe.fit(df.clean_desc, df.category); joblib.dump(pipe, ROOT / "models" / "categorizer.joblib")
    df["predicted_category"] = pipe.predict(df.clean_desc)
    df.to_sql("clean_transactions", con, if_exists="replace", index=False)

if __name__ == "__main__":
    main()
