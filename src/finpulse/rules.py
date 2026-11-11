"""Keyword rules used to seed labels when a statement has no categories yet (cold start).

Deliberately small and generic: well-known Indian merchants plus category words. They label only the
rows they recognise; a classifier trained on those rows then labels the rest, and low-confidence rows go
to a review queue. These rules have only been tested on simulated data, not on real statements.
"""
KEYWORDS = {
    "Food Delivery": ["SWIGGY", "ZOMATO", "DOMINOS", "MCDONALD", "KFC", "STARBUCKS", "CAFE", "RESTAURANT", "PIZZA",
                      "BURGER", "BIRYANI", "BAKERY", "FAASOS", "SUBWAY"],
    "Groceries": ["BIGBASKET", "DMART", "BLINKIT", "ZEPTO", "GROCERY", "SUPERMARKET", "KIRANA", "HYPERMARKET",
                  "VEGETABLE", "FRUIT", "DAIRY", "JIOMART", "RELIANCE FRESH"],
    "Transport": ["UBER", "OLA", "RAPIDO", "IRCTC", "REDBUS", "METRO", "FASTAG", "FUEL", "PETROL", "CAB", "TAXI",
                  "INDIAN OIL", "BHARAT PETROLEUM"],
    "Rent": ["RENT", "LANDLORD", "MAINT"],
    "Utilities": ["ELECTRICITY", "WATER BILL", "GAS", "BROADBAND", "FIBERNET", "RECHARGE", "POSTPAID", "DTH",
                  "TATA PLAY", "TATA POWER"],
    "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA", "DECATHLON", "CROMA", "STORE", "MART", "MEESHO"],
    "Entertainment": ["NETFLIX", "SPOTIFY", "HOTSTAR", "BOOKMYSHOW", "PVR", "INOX", "CINEMA", "YOUTUBE", "PRIME",
                      "GAME", "CONCERT"],
    "Health": ["PHARMACY", "PHARMEASY", "APOLLO", "HOSPITAL", "CLINIC", "DENTAL", "PATHLAB", "DIAGNOSTIC", "MEDICAL",
               "CULT FIT", "GYM", "NETMEDS"],
    "Investment": ["SIP", "MUTUAL FUND", "ZERODHA", "GROWW", "PPF", "NPS"],
    "Income": ["SALARY", "INTEREST CREDIT", "FREELANCE"],
}


def seed_label(clean_desc: str, txn_type: str = "debit"):
    """Return a category if a keyword matches, else None. Credits that match nothing are treated as Income."""
    for cat, words in KEYWORDS.items():
        if any(w in clean_desc for w in words):
            return cat
    return "Income" if txn_type == "credit" else None
