from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "finpulse.db"
SEED = 7
START = "2023-07-01"; N_MONTHS = 36
CATEGORIES = ["Food Delivery", "Groceries", "Transport", "Rent", "Utilities", "Shopping",
              "Entertainment", "Health", "Investment", "Income"]
DISCRETIONARY = ["Food Delivery", "Shopping", "Entertainment"]
