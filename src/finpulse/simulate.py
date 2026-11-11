"""Synthetic personal-finance data: a few years of bank-statement rows for one simulated user.

What makes the simulator more than random strings:
  * messy descriptions (UPI/POS prefixes, reference numbers, city codes);
  * ~120 merchants, some of which first appear late in the history, so a time-based test set
    contains merchants the model has never seen;
  * a controllable share of "descriptive" merchants whose names contain a category word
    (CITY PHARMACY, SHELL FUEL STATION), as many real statement strings do;
  * ambiguous merchants (AMAZON, DMART, BLINKIT) and a few percent of wrong labels;
  * recurring subscriptions, rent, SIP and salary on a monthly schedule (ground truth for the
    subscription detector) and injected unusually large purchases (ground truth for the
    anomaly detector).

Ground truth lives in a separate table so no pipeline stage can read it by accident.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import SimConfig, CATEGORIES

BRANDS = {
    "Food Delivery": ["SWIGGY", "ZOMATO", "DOMINOS PIZZA", "MCDONALDS", "KFC", "STARBUCKS", "CAFE COFFEE DAY",
                      "BURGER KING", "SUBWAY", "BARBEQUE NATION", "HALDIRAMS", "FAASOS"],
    "Groceries": ["BIGBASKET", "DMART", "RELIANCE FRESH", "ZEPTO", "BLINKIT", "SPENCERS", "MORE RETAIL",
                  "NATURES BASKET", "JIOMART", "STAR BAZAAR"],
    "Transport": ["UBER", "OLA CABS", "IRCTC", "INDIAN OIL", "HP PETROL PUMP", "DELHI METRO", "REDBUS",
                  "FASTAG", "RAPIDO", "BHARAT PETROLEUM"],
    "Utilities": ["AIRTEL POSTPAID", "JIO RECHARGE", "BSES ELECTRICITY", "TATA PLAY", "MAHANAGAR GAS", "VI RECHARGE",
                  "TATA POWER", "INDRAPRASTHA GAS"],
    "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA", "DECATHLON", "CROMA", "LIFESTYLE STORE",
                 "IKEA", "PUMA", "H AND M", "MEESHO"],
    "Entertainment": ["BOOKMYSHOW", "PVR CINEMAS", "STEAM GAMES", "INOX", "PLAYSTATION STORE", "SONYLIV", "ZEE5"],
    "Health": ["APOLLO PHARMACY", "PHARMEASY", "ONE MG", "MAX HOSPITAL", "DR LAL PATHLABS", "NETMEDS",
               "FORTIS HOSPITAL"],
}
DESCRIPTIVE = {
    "Food Delivery": ["TASTY BITES RESTAURANT", "URBAN CAFE", "SPICE GARDEN RESTAURANT", "PIZZA HOUSE", "BURGER JOINT",
                      "CHAI POINT CAFE", "BIRYANI HOUSE", "SWEET SHOP BAKERY"],
    "Groceries": ["FRESH SUPERMARKET", "VEGETABLE MART", "DAILY NEEDS GROCERY", "ORGANIC KIRANA STORE",
                  "CITY HYPERMARKET", "MILK DAIRY BOOTH", "FRUIT MARKET"],
    "Transport": ["CITY CAB SERVICE", "QUICK TAXI", "SHELL FUEL STATION", "HIGHWAY PETROL PUMP",
                  "METRO CARD RECHARGE", "AUTO RICKSHAW", "BUS TICKET BOOKING"],
    "Utilities": ["STATE ELECTRICITY BOARD", "CITY WATER BILL", "PIPED GAS BILL", "MOBILE POSTPAID BILL",
                  "DTH RECHARGE", "BROADBAND BILL PAYMENT"],
    "Shopping": ["FASHION STORE", "ELECTRONICS MART", "SHOE GALLERY", "BOOK DEPOT", "HOME DECOR STORE",
                 "GADGET WORLD", "TOY SHOP"],
    "Entertainment": ["SKY CINEMAS", "GAME ZONE ARCADE", "CONCERT TICKETS", "BOWLING ALLEY", "THEATRE BOOKING",
                      "AMUSEMENT PARK"],
    "Health": ["CITY PHARMACY", "WELLNESS CLINIC", "DENTAL CARE CENTRE", "EYE HOSPITAL", "DIAGNOSTIC LAB",
               "MEDICAL STORE"],
}
AMBIG = {"AMAZON": ["Shopping", "Groceries", "Entertainment"], "DMART": ["Groceries", "Shopping"],
         "BLINKIT": ["Groceries", "Food Delivery"], "JIOMART": ["Groceries", "Shopping"]}
# monthly recurring items: (merchant, category, amount range). Each user gets a random subset.
SUBSCRIPTIONS = [("NETFLIX", "Entertainment", (149, 649)), ("SPOTIFY", "Entertainment", (59, 119)),
                 ("YOUTUBE PREMIUM", "Entertainment", (129, 189)), ("HOTSTAR", "Entertainment", (149, 299)),
                 ("CULT FIT", "Health", (1500, 2500)), ("ACT FIBERNET", "Utilities", (899, 1299)),
                 ("AMAZON PRIME", "Entertainment", (125, 299))]
AMT = {"Food Delivery": (180, 90), "Groceries": (650, 300), "Transport": (260, 180), "Utilities": (700, 350),
       "Shopping": (1700, 1200), "Entertainment": (420, 250), "Health": (520, 400)}
FREQ = {"Food Delivery": 9, "Groceries": 7, "Transport": 10, "Utilities": 2.2, "Shopping": 3.2,
        "Entertainment": 1.6, "Health": 1.0}
DISCRETIONARY_SEASON = {10: 1.35, 11: 1.4, 12: 1.3, 1: 1.0, 6: .9, 7: .9}
PREFIX = ["UPI-", "POS ", "UPI/", "NEFT-", "IMPS-", "", "ECOM "]
CITY = ["DELHI", "GURGAON", "NOIDA", "BLR", "MUMBAI", ""]


@dataclass
class Sim:
    raw: pd.DataFrame      # what a bank statement shows: txn_id, date, description, amount, type, category
    truth: pd.DataFrame    # evaluation only: txn_id, merchant, is_recurring, is_anomaly
    investments: pd.DataFrame


def _describe(rng, merchant):
    return f"{rng.choice(PREFIX)}{merchant} {rng.integers(10000, 99999)} {rng.choice(CITY)}".strip()


def _split_pool(rng, names, late_frac):
    """Return (early, all): merchants usable from the start vs. everything, with `late_frac` held back."""
    names = list(rng.permutation(names))
    n_late = min(int(round(late_frac * len(names))), len(names) - 1)
    return names[: len(names) - n_late], names


def simulate(cfg: SimConfig = SimConfig()) -> Sim:
    rng = np.random.default_rng(cfg.seed)
    months = pd.date_range(cfg.start, periods=cfg.n_months, freq="MS")
    late_from = int(cfg.n_months * .75)
    pools = {c: (_split_pool(rng, BRANDS[c], cfg.new_merchant_frac), _split_pool(rng, DESCRIPTIVE[c], cfg.new_merchant_frac))
             for c in BRANDS}
    subs = [s for s in SUBSCRIPTIONS if rng.random() < .7] or SUBSCRIPTIONS[:2]
    sub_amt = {s[0]: round(float(rng.uniform(*s[2])), 2) for s in subs}
    rent, sip, salary = (round(float(rng.uniform(*r)), 2) for r in ((14000, 26000), (3000, 8000), (48000, 95000)))
    rent_name = str(rng.choice(["RENT PAYMENT LANDLORD", "NOBROKER RENT", "HOUSING SOCIETY MAINT"]))
    sip_name = str(rng.choice(["ZERODHA SIP", "GROWW MF SIP", "NIPPON INDEX FUND SIP"]))
    rows, truth = [], []

    def add(date, merchant, amount, typ, label, recurring=False):
        rows.append((date, _describe(rng, merchant), round(float(amount), 2), typ, label))
        truth.append((merchant, recurring))

    for mi, m in enumerate(months):
        trend = 1 + .012 * mi
        d = lambda day: (m + pd.Timedelta(days=int(np.clip(day, 1, 28)) - 1))
        add(d(1), "SALARY CREDIT ACME CORP", salary, "credit", "Income", True)
        add(d(1 + rng.integers(0, 3)), rent_name, rent, "debit", "Rent", True)
        add(d(5 + rng.integers(0, 3)), sip_name, sip, "debit", "Investment", True)
        for name, cat, _ in subs:
            add(d(8 + sum(map(ord, name)) % 15 + rng.integers(-1, 2)), name, sub_amt[name], "debit", cat, True)
        for cat, lam in FREQ.items():
            mult = DISCRETIONARY_SEASON.get(m.month, 1.0) if cat in ("Food Delivery", "Shopping", "Entertainment") else 1.0
            for _ in range(rng.poisson(lam * mult)):
                (b_early, b_all), (d_early, d_all) = pools[cat]
                use_desc = rng.random() < cfg.descriptive_frac
                avail = (d_all if use_desc else b_all) if mi >= late_from else (d_early if use_desc else b_early)
                merchant = str(rng.choice(avail))
                mu, sd = AMT[cat]
                amount = max(20, rng.normal(mu * trend, sd))
                label = cat
                if merchant in AMBIG and rng.random() < .35:
                    label = str(rng.choice(AMBIG[merchant]))
                if rng.random() < cfg.label_noise:
                    label = str(rng.choice([c for c in AMT if c != cat]))
                add(d(rng.integers(1, 29)), merchant, amount, "debit", label)

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "type", "category"])
    tr = pd.DataFrame(truth, columns=["merchant", "is_recurring"])
    df["date"] = df.date.dt.date.astype(str)
    order = np.argsort(df.date.to_numpy(), kind="stable")
    df, tr = df.iloc[order].reset_index(drop=True), tr.iloc[order].reset_index(drop=True)
    df.insert(0, "txn_id", range(1, len(df) + 1)); tr.insert(0, "txn_id", df.txn_id)
    # injected anomalies: ordinary purchases made 5-12x larger than normal, after the first half year
    ordinary = np.flatnonzero((df.type == "debit") & ~tr.is_recurring & (df.date >= str(months[6].date())))
    hit = rng.choice(ordinary, size=min(cfg.n_anomalies, len(ordinary)), replace=False)
    df.loc[hit, "amount"] = (df.loc[hit, "amount"] * rng.uniform(5, 12, size=len(hit))).round(2)
    tr["is_anomaly"] = False; tr.loc[hit, "is_anomaly"] = True
    val, inv = 0.0, []
    for m in months:
        val = (val + sip) * (1 + rng.normal(.009, .035)); inv.append((m.date().isoformat(), sip, round(val, 2)))
    return Sim(df, tr, pd.DataFrame(inv, columns=["month", "sip_amount", "portfolio_value"]))
