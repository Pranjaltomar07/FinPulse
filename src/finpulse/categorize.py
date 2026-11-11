"""Expense categoriser with confidence, a review queue, and a learn-from-corrections loop."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def make_model(C: float = 10):
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=C))


class Categoriser:
    """Character n-gram TF-IDF + logistic regression. `predict` returns a label and a confidence."""
    def __init__(self, C: float = 10):
        self.C = C

    def fit(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)
        self.model = make_model(self.C).fit(self.texts, self.labels)
        return self

    def predict(self, texts):
        p = self.model.predict_proba(list(texts))
        return self.model.classes_[p.argmax(axis=1)], p.max(axis=1)

    def learn(self, new_texts, new_labels):
        """Add user-corrected rows and refit (the correction loop behind the review queue)."""
        return Categoriser(self.C).fit(self.texts + list(new_texts), self.labels + list(new_labels))


def lookup_predict(train_texts, train_labels, texts):
    """Exact-match table: most common label for each cleaned string, majority class if unseen."""
    tr = pd.DataFrame({"t": list(train_texts), "y": list(train_labels)})
    lut = tr.groupby("t").y.agg(lambda s: s.mode()[0])
    return pd.Series(list(texts)).map(lut).fillna(tr.y.mode()[0]).to_numpy()


def nn_predict(train_texts, train_labels, texts):
    """Fuzzy match: label of the most similar training string by character-trigram cosine similarity."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3)).fit(list(train_texts))
    sim = (vec.transform(list(texts)) @ vec.transform(list(train_texts)).T).toarray()
    return np.asarray(list(train_labels))[sim.argmax(axis=1)]


def corrupt(s: str, rng: np.random.Generator) -> str:
    """Mangle a description like a real statement: drop, swap or replace a character, or truncate."""
    if len(s) < 5:
        return s
    chars = list(s)
    for _ in range(int(rng.integers(1, 3))):
        op = rng.choice(["drop", "swap", "sub", "trunc"]); i = int(rng.integers(1, len(chars) - 1))
        if op == "drop": del chars[i]
        elif op == "swap": chars[i], chars[i - 1] = chars[i - 1], chars[i]
        elif op == "sub": chars[i] = chr(int(rng.integers(65, 91)))
        elif op == "trunc" and len(chars) > 6: chars = chars[: len(chars) - int(rng.integers(1, 4))]
    out = "".join(chars)
    return out if out != s else s[:-1]


def accuracy(y, p) -> float:
    return float((np.asarray(y) == np.asarray(p)).mean())


def risk_coverage(y, pred, conf, coverages=(1.0, .9, .8, .7, .6, .5)) -> dict:
    """Accuracy among the most confident `coverage` share of predictions (the rest would go to review)."""
    order = np.argsort(-np.asarray(conf), kind="stable"); y, pred = np.asarray(y)[order], np.asarray(pred)[order]
    return {c: accuracy(y[: max(1, int(round(c * len(y))))], pred[: max(1, int(round(c * len(y))))]) for c in coverages}


def active_learning(train_texts, train_labels, test_texts, test_labels, seed, n0=60, step=60, rounds=7):
    """Labelling budget curve: accuracy vs. number of labelled rows, random vs. least-confident sampling."""
    rng = np.random.default_rng(seed); T, Y = np.asarray(list(train_texts)), np.asarray(list(train_labels))
    start = rng.choice(len(T), n0, replace=False); curves = {}
    for name in ("random", "uncertainty"):
        lab = list(start); rest = set(range(len(T))) - set(lab); acc = []
        r2 = np.random.default_rng(seed + 1)
        for r in range(rounds):
            m = Categoriser().fit(T[lab], Y[lab]); acc.append((len(lab), accuracy(test_labels, m.predict(test_texts)[0])))
            if r == rounds - 1: break
            pool = np.array(sorted(rest))
            pick = (r2.choice(pool, step, replace=False) if name == "random"
                    else pool[np.argsort(m.predict(T[pool])[1], kind="stable")[:step]])
            lab += list(pick); rest -= set(pick)
        curves[name] = acc
    return curves
