"""Shared constants and the simulator configuration."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ["Food Delivery", "Groceries", "Transport", "Rent", "Utilities", "Shopping",
              "Entertainment", "Health", "Investment", "Income"]
FIXED = ("Rent", "Income", "Investment")                      # not ordinary day-to-day spend
SPEND_CATS = [c for c in CATEGORIES if c not in ("Income", "Investment")]
DISCRETIONARY = ["Food Delivery", "Shopping", "Entertainment"]


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    n_months: int = 36
    start: str = "2023-07-01"
    descriptive_frac: float = 0.35   # share of merchants whose name contains a category word (PHARMACY, FUEL...)
    new_merchant_frac: float = 0.25  # share of merchants that first appear in the final quarter of the history
    label_noise: float = 0.03        # share of ordinary-spend labels set to a wrong category
    n_anomalies: int = 10            # injected unusually large purchases (ground truth for the detector)
