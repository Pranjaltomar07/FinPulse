# Findings - FinPulse (simulation)

## What this is
A pipeline for personal-finance analytics: messy bank-statement strings -> cleaned analytics tables -> expense categoriser -> spend forecast -> net-worth view -> savings what-if. The transactions are simulated (1,434 rows over 36 months), so conclusions about spending behaviour are not claimed.

## What the experiments showed
- **Categoriser:** char n-gram TF-IDF + logistic regression reaches ~95% accuracy on a time-based holdout, but an exact-match lookup table scores the same. On simulated merchants the lookup is already sufficient; the model's advantage would appear on unseen real merchant strings, which this data cannot test.
- **Forecast:** a Random Forest does not beat a 3-month moving average for total monthly spend (WAPE 20.6% vs 13.2%; naive last-month 15.6%). With 36 points of mostly random noise there is little to learn. Reported as a negative result.
- **Savings what-if:** a 10% cut in discretionary categories saves about Rs 14k/year (1.8% of income) on the simulated user. A scenario calculation, not a prediction.

## Bugs found and fixed during development
- Label noise randomly relabelled salary/rent rows as small-spend categories, creating a Rs 68k outlier month that wrecked the Transport forecast. Noise is now restricted to ordinary spend (tested).
- Net worth double-counted SIP contributions (cash was not reduced by investments). Fixed and covered by a test.
- MAPE exploded on near-zero months (28,000%); replaced by WAPE.

## Limitations
Simulated data; one synthetic user; categories follow merchant names the simulator wrote. Not suitable for claims about real behaviour.
