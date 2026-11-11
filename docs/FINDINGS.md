# Findings

Every number is in `results/summary.csv` (mean, 95% t-interval over 10 simulated users). `python -m finpulse evaluate` reproduces them.

## What held up
1. **Typos:** the n-gram model keeps 83.7% accuracy; exact lookup falls to 25.5%; fuzzy nearest-neighbour 76.4%.
2. **Confidence is informative:** accuracy rises from 87.6% (everything) to 97.3% (top 80% by confidence) and 98.6% (top 50%).
3. **Subscription detection** is perfect on regular simulated payments (a weak test; see below).
4. **Per-category anomaly scoring** beats one global score by 65 F1 points.
5. **Ensemble forecast** beats naive by 5.0 WAPE points (3.6 to 6.4) and the 3-month average by 1.5.

## What did not
1. **New merchants.** 52.4% on unseen merchants in later months; 36.2% when all test merchants are held out. This is the main limit.
2. **Active learning** by least confidence lost to random sampling (70.7% vs 80.8% at 180 labels).
3. **Forecast band** covers 72%, not 80%, and is about 43% wide.
4. **Random forest** is not clearly better than a moving average (difference -1.0, interval -2.6 to 0.6).
5. **Anomaly recall** is 69%; at a stricter threshold precision rises and recall falls (`results/anomaly_sweep.csv`).

## Why subscriptions look too good
Simulated subscriptions are charged on near-identical days for an identical amount. Real ones drift (price rises, weekends, failed payments). Treat 100% as "the code works", not "the problem is solved".

## Next steps
Run `analyze` on a real anonymised statement; add merchant-name enrichment (a lookup of public merchant categories) for unseen merchants; calibrate the forecast band.
