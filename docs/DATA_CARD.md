# Data card: simulated statements

**Source:** `src/finpulse/simulate.py`, seeded. Users 100 to 109 are used for evaluation; users 200 to 204 for the sensitivity study.
**Size per user:** 36 months, about 1,700 transactions, about 120 merchant strings.
**Columns shown to the pipeline:** txn_id, date, description, amount, type, category (the user's own label).
**Hidden:** a truth table with merchant, is_recurring, is_anomaly. Stored separately and used only for scoring.

## Realism features
Channel prefixes (UPI, POS, NEFT), reference numbers, city codes; merchants that first appear late; ambiguous merchants (AMAZON, DMART, BLINKIT) with mixed labels; 3% wrong labels; seasonality in discretionary spend; a mild upward trend.

## Design choices that affect results
- **Descriptive names:** 35% of merchants have a name containing a category word. Changing it to 0% or 70% shifts unseen-merchant accuracy (see `results/sensitivity.csv`).
- **New merchants:** 25% of each pool appears only after month 27.
- **Anomalies:** 10 purchases, 5 to 12 times larger than normal, after month 6. Real anomalies are rarely that clean.
- **Subscriptions:** exactly regular. Easier than real life.

## Not included
No real people, accounts or merchants' real prices. Merchant names are real brands used only as strings. Not financial advice.
