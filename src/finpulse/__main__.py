"""Command line:  python -m finpulse demo | evaluate | analyze --csv FILE --date-col ... --desc-col ..."""
import argparse
from pathlib import Path
from .config import ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(prog="finpulse"); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="simulate a user and build the dashboard"); d.add_argument("--out", default=str(ROOT / "output"))
    sub.add_parser("evaluate", help="10-user evaluation (about 1 minute)")
    a = sub.add_parser("analyze", help="run on your own statement CSV (stays on your machine)")
    a.add_argument("--csv", required=True); a.add_argument("--date-col", required=True); a.add_argument("--desc-col", required=True)
    a.add_argument("--amount-col"); a.add_argument("--debit-col"); a.add_argument("--credit-col"); a.add_argument("--category-col"); a.add_argument("--out", default=str(ROOT / "output"))
    a.add_argument("--monthfirst", action="store_true", help="dates are MM/DD/YYYY")
    n = ap.parse_args(argv)
    if n.cmd == "evaluate":
        from . import evaluate, figures
        evaluate.main(); figures.make_all(ROOT / "results", ROOT / "docs" / "img"); return
    from . import pipeline, dashboard
    if n.cmd == "demo":
        from .simulate import simulate
        raw = simulate().raw; res = pipeline.run(raw, Path(n.out))
    else:
        from .ingest import load_statement
        raw = load_statement(n.csv, n.date_col, n.desc_col, n.amount_col, n.debit_col, n.credit_col, n.category_col, dayfirst=not n.monthfirst)
        res = pipeline.run(raw, Path(n.out), cold="category" not in raw.columns)
    dashboard.render(res, Path(n.out) / "dashboard.html"); print(f"Wrote {Path(n.out) / 'dashboard.html'} and finpulse.db")


if __name__ == "__main__":
    main()
