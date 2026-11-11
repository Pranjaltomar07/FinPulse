"""Ten simulated users, every claim measured with a 95% t-interval. Run: python -m finpulse evaluate"""
import dataclasses, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from .config import SimConfig, ROOT
from .simulate import simulate
from .clean import prepare
from . import categorize as C, recurring as R, anomaly as A, forecast as F

Z_THRESHOLD = 4.0   # fixed in advance of the 10-user run, not tuned per user


def ci(x):
    x = np.asarray(x, float); n = len(x); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def user(seed: int, **over):
    cfg = dataclasses.replace(SimConfig(), seed=seed, **over); s = simulate(cfg); return prepare(s.raw), s.truth, cfg


def categoriser_one(df, seed, cfg):
    rng = np.random.default_rng(seed); o = {}
    cut = sorted(df.month.unique())[int(cfg.n_months * .75)]; a, b = df[df.month < cut], df[df.month >= cut]
    unseen = ~b.clean_desc.isin(set(a.clean_desc)); o["unseen_share"] = float(unseen.mean())
    m = C.Categoriser().fit(a.clean_desc, a.category); p, conf = m.predict(b.clean_desc)
    o["time_model"] = C.accuracy(b.category, p); o["time_unseen_only"] = C.accuracy(b.category[unseen], p[unseen]) if unseen.any() else np.nan
    o["time_lookup"] = C.accuracy(b.category, C.lookup_predict(a.clean_desc, a.category, b.clean_desc))
    o["time_majority"] = C.accuracy(b.category, np.repeat(a.category.mode()[0], len(b)))
    messy = b.clean_desc.map(lambda s: C.corrupt(s, rng))
    o["messy_model"] = C.accuracy(b.category, m.predict(messy)[0])
    o["messy_lookup"] = C.accuracy(b.category, C.lookup_predict(a.clean_desc, a.category, messy))
    o["messy_nn"] = C.accuracy(b.category, C.nn_predict(a.clean_desc, a.category, messy))
    for c, v in C.risk_coverage(b.category, p, conf).items(): o[f"rc_{int(c*100)}"] = v
    sc = {k: [] for k in ("model", "lookup", "nn", "majority")}
    for tri, tei in GroupKFold(5).split(df, groups=df.clean_desc):
        x, y = df.iloc[tri], df.iloc[tei]; assert set(x.clean_desc).isdisjoint(y.clean_desc)
        sc["model"].append(C.accuracy(y.category, C.Categoriser().fit(x.clean_desc, x.category).predict(y.clean_desc)[0]))
        sc["lookup"].append(C.accuracy(y.category, C.lookup_predict(x.clean_desc, x.category, y.clean_desc)))
        sc["nn"].append(C.accuracy(y.category, C.nn_predict(x.clean_desc, x.category, y.clean_desc)))
        sc["majority"].append(C.accuracy(y.category, np.repeat(x.category.mode()[0], len(y))))
    for k, v in sc.items(): o[f"heldout_{k}"] = float(np.mean(v))
    al = C.active_learning(a.clean_desc, a.category, b.clean_desc, b.category, seed)
    for k, v in al.items():
        for n, acc in v: o[f"al_{k}_{n}"] = acc
    return o


def rest_one(df, tr, seed):
    o = {}; found = R.detect_recurring(df); o.update({f"rec_{k}": v for k, v in A.prf(tr.is_recurring, R.flag_rows(df, found)).items()})
    d = df.assign(category=df.category)
    for name, by in (("anom_cat", True), ("anom_global", False)):
        o.update({f"{name}_{k}": v for k, v in A.prf(tr.is_anomaly, A.score(d, by) > Z_THRESHOLD).items()})
    T = F.monthly_totals(df); act, P = F.rolling_origin(T.TOTAL, seed)
    for k, v in P.items(): o[f"wape_{k}"] = F.wape(act, v)
    o.update({f"int_{k}": v for k, v in F.interval_backtest(act, P["ensemble"]).items()}); return o


def main(n_users: int = 10, out=None):
    out = out or ROOT / "results"; out.mkdir(exist_ok=True); rows = []
    for s in range(n_users):
        df, tr, cfg = user(100 + s); rows.append({"user": s, **categoriser_one(df, s, cfg), **rest_one(df, tr, s)}); print("user", s, flush=True)
    U = pd.DataFrame(rows); U.round(4).to_csv(out / "per_user.csv", index=False)
    R_ = []
    for col in U.columns[1:]:
        m, lo, hi = ci(U[col] * (1 if col.startswith(("wape", "int_width")) or col == "unseen_share" and False else 1)); R_.append({"metric": col, "mean": m, "ci_lo": lo, "ci_hi": hi})
    pairs = [("time_model", "time_lookup"), ("heldout_model", "heldout_lookup"), ("heldout_model", "heldout_majority"), ("heldout_nn", "heldout_model"),
             ("messy_model", "messy_lookup"), ("messy_nn", "messy_model"), ("anom_cat_f1", "anom_global_f1"),
             ("wape_ensemble", "wape_ma3"), ("wape_ensemble", "wape_seasonal_naive"), ("wape_rf", "wape_ma3"), ("wape_ensemble", "wape_naive")]
    for a, b in pairs:
        m, lo, hi = ci(U[a] - U[b]); R_.append({"metric": f"{a} minus {b}", "mean": m, "ci_lo": lo, "ci_hi": hi})
    S = pd.DataFrame(R_); S.round(4).to_csv(out / "summary.csv", index=False)
    # sensitivity to how many merchants have descriptive names (the simulator's biggest design choice)
    sens = []
    for frac in (0.0, 0.35, 0.7):
        v = []
        for s in range(5):
            df, tr, cfg = user(200 + s, descriptive_frac=frac)
            cut = sorted(df.month.unique())[int(cfg.n_months * .75)]; a, b = df[df.month < cut], df[df.month >= cut]
            un = ~b.clean_desc.isin(set(a.clean_desc)); p = C.Categoriser().fit(a.clean_desc, a.category).predict(b.clean_desc)[0]
            lk = C.lookup_predict(a.clean_desc, a.category, b.clean_desc)
            v.append((C.accuracy(b.category[un], p[un]), C.accuracy(b.category[un], lk[un])))
        v = np.array(v); sens.append({"descriptive_frac": frac, "model_unseen_acc": v[:, 0].mean(), "lookup_unseen_acc": v[:, 1].mean()})
    pd.DataFrame(sens).round(4).to_csv(out / "sensitivity.csv", index=False)
    # active learning + risk-coverage + anomaly threshold sweep, pooled
    sweep = []
    for z in (3, 3.5, 4, 5):
        for name, by in (("per_category", True), ("global", False)):
            r = [A.prf(tr_.is_anomaly, A.score(df_, by) > z) for df_, tr_, _ in (user(100 + s) for s in range(10))]
            sweep.append({"z": z, "method": name, **{k: float(np.mean([x[k] for x in r])) for k in ("precision", "recall", "f1")}})
    pd.DataFrame(sweep).round(4).to_csv(out / "anomaly_sweep.csv", index=False)
    return U, S


if __name__ == "__main__":
    main()
