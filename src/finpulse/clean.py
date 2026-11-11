"""Turn raw bank-statement strings into a clean merchant key."""
import re
import pandas as pd

CHANNELS = re.compile(r"\b(UPI|POS|NEFT|IMPS|ECOM|ATM|NACH|ACH|RTGS)\b[-/ ]?")
CITIES = re.compile(r"\b(DELHI|GURGAON|NOIDA|BLR|BANGALORE|MUMBAI|PUNE|CHENNAI|KOLKATA|HYDERABAD)\b")


def clean_desc(s: str) -> str:
    """'UPI-SWIGGY 48213 DELHI' -> 'SWIGGY'. Removes channel prefixes, reference numbers and city codes."""
    s = CHANNELS.sub(" ", str(s).upper())
    s = re.sub(r"\d{4,}", " ", s)
    s = CITIES.sub(" ", s)
    return " ".join(re.sub(r"[^A-Z ]", " ", s).split())


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Add clean_desc and month columns to a standard statement frame."""
    df = raw.copy()
    df["clean_desc"] = df.description.map(clean_desc)
    df["month"] = df.date.astype(str).str[:7]
    return df
