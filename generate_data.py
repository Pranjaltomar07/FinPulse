"""Synthetic personal-finance data: ~3 years of bank statement rows with messy merchant
strings (UPI/POS prefixes, ref numbers, city codes), plus monthly SIP investments.
Some merchants are deliberately ambiguous (e.g. AMAZON, DMART) and ~3% of labels are
noisy, so categorisation accuracy is not artificially 100%."""
import numpy as np, pandas as pd, sqlite3
from config import *

MERCH = {
 "Food Delivery": ["SWIGGY", "ZOMATO", "DOMINOS PIZZA", "MCDONALDS", "KFC", "BLINKIT CAFE", "STARBUCKS", "CAFE COFFEE DAY"],
 "Groceries": ["BIGBASKET", "DMART", "RELIANCE FRESH", "ZEPTO", "BLINKIT", "SPENCERS", "LOCAL KIRANA"],
 "Transport": ["UBER", "OLA CABS", "IRCTC", "INDIAN OIL FUEL", "HP PETROL PUMP", "DELHI METRO", "REDBUS", "FASTAG"],
 "Rent": ["RENT PAYMENT LANDLORD", "NOBROKER RENT", "HOUSING SOCIETY MAINT"],
 "Utilities": ["AIRTEL POSTPAID", "JIO RECHARGE", "BSES ELECTRICITY", "TATA PLAY", "MAHANAGAR GAS", "ACT FIBERNET"],
 "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA", "DECATHLON", "CROMA", "LIFESTYLE STORE"],
 "Entertainment": ["NETFLIX", "SPOTIFY", "BOOKMYSHOW", "PVR CINEMAS", "YOUTUBE PREMIUM", "STEAM GAMES", "HOTSTAR"],
 "Health": ["APOLLO PHARMACY", "PHARMEASY", "1MG", "MAX HOSPITAL", "CULT FIT", "DR LAL PATHLABS"],
 "Investment": ["ZERODHA SIP", "GROWW MF SIP", "NIPPON INDEX FUND SIP", "PPF DEPOSIT"],
 "Income": ["SALARY CREDIT ACME CORP", "FREELANCE PAYMENT NEFT", "INTEREST CREDIT"],
}
AMBIG = {"AMAZON": ["Shopping", "Groceries", "Entertainment"], "DMART": ["Groceries", "Shopping"],
         "BLINKIT": ["Groceries", "Food Delivery"], "BLINKIT CAFE": ["Food Delivery", "Groceries"]}
AMT = {"Food Delivery": (180, 90), "Groceries": (650, 300), "Transport": (260, 180), "Rent": (18000, 0),
       "Utilities": (700, 350), "Shopping": (1700, 1200), "Entertainment": (420, 250), "Health": (520, 400),
       "Investment": (5000, 0), "Income": (65000, 0)}
FREQ = {"Food Delivery": 9, "Groceries": 7, "Transport": 10, "Rent": 1, "Utilities": 3, "Shopping": 3.2,
        "Entertainment": 2.5, "Health": 1.2, "Investment": 1, "Income": 1}
PREFIX = ["UPI-", "POS ", "UPI/", "NEFT-", "IMPS-", "", "ECOM "]
CITY = ["DELHI", "GURGAON", "NOIDA", "BLR", "MUMBAI", ""]
SEASON = {10: 1.35, 11: 1.4, 12: 1.3, 1: 1.0, 6: .9, 7: .9}  # festive / sale months raise discretionary spend

def main():
    rng = np.random.default_rng(SEED)
    rows = []; months = pd.date_range(START, periods=N_MONTHS, freq="MS")
    for mi, m in enumerate(months):
        trend = 1 + .012 * mi
        for cat, lam in FREQ.items():
            mult = SEASON.get(m.month, 1.0) if cat in DISCRETIONARY else 1.0
            n = 1 if cat in ("Rent", "Income", "Investment") else rng.poisson(lam * mult)
            for _ in range(n):
                mu, sd = AMT[cat]
                amt = max(20, rng.normal(mu * (trend if cat not in ("Rent", "Income", "Investment") else 1), sd))
                day = 1 if cat in ("Rent", "Income") else rng.integers(1, 29)
                m_name = rng.choice(MERCH[cat])
                label = cat
                if m_name in AMBIG and rng.random() < .35: label = rng.choice(AMBIG[m_name])
                if cat not in ("Income", "Rent", "Investment") and rng.random() < .03:
                    label = rng.choice([c for c in CATEGORIES if c not in (cat, "Income", "Rent", "Investment")])
                desc = f"{rng.choice(PREFIX)}{m_name} {rng.integers(10000, 99999)} {rng.choice(CITY)}".strip()
                rows.append((m + pd.Timedelta(days=int(day) - 1), desc, round(float(amt), 2),
                             "credit" if cat == "Income" else "debit", label))
    df = pd.DataFrame(rows, columns=["date", "description", "amount", "type", "category"]).sort_values("date")
    df["date"] = df.date.dt.date.astype(str); df.insert(0, "txn_id", range(1, len(df) + 1))
    # investment valuations: monthly SIP with random market returns
    val, inv = 0, []
    for m in months:
        val = (val + 5000) * (1 + rng.normal(.009, .035)); inv.append((m.date().isoformat(), 5000, round(val, 2)))
    con = sqlite3.connect(DB_PATH)
    df.to_sql("raw_transactions", con, if_exists="replace", index=False)
    pd.DataFrame(inv, columns=["month", "sip_amount", "portfolio_value"]).to_sql("raw_investments", con, if_exists="replace", index=False)
    con.close(); print(f"{len(df)} transactions over {N_MONTHS} months")

if __name__ == "__main__":
    DB_PATH.parent.mkdir(exist_ok=True); main()
