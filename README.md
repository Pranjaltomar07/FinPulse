# FinPulse - Personal Finance Intelligence (simulation)

[![tests](https://github.com/YOUR-USERNAME/finpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranjaltomar07/finpulse/actions)

*Personal finance analytics pipeline on simulated data (categoriser, forecast, what-if).*

> **All data here is simulated.** This is an engineering exercise in building a personal-finance analytics pipeline, not a study of real spending. The metrics describe my simulator, not real people. See [FINDINGS.md](FINDINGS.md).

Bank-statement-style transactions -> cleaning/ETL -> ML expense categoriser -> spend forecasting -> net worth view -> savings what-if -> Streamlit dashboard.

**Data is synthetic** (`src/generate_data.py`): ~1,450 transactions over 36 months with messy merchant strings, deliberately ambiguous merchants (AMAZON, DMART, BLINKIT) and ~3% label noise. Point `raw_transactions` at a real statement export (date, description, amount, type) to use your own data.

## Run
```bash
pip install -r requirements.txt
python run_all.py
streamlit run app.py
pytest -q tests
```

## Pipeline
| Stage | File |
|---|---|
| Generate / ingest | `src/generate_data.py` |
| Clean + analytics tables (monthly_summary, net_worth) | `src/etl.py` |
| Categoriser: char n-gram TF-IDF + logistic regression, time-based split | `src/categorize.py` |
| Forecast: Random Forest on lags + seasonality vs naive baselines | `src/forecast.py` |
| Savings what-if on last 12 months | `src/savings.py` |
| Dashboard | `app.py` |

SQLite stands in for PostgreSQL; `run_all.py` for an Airflow DAG.

## Notes and limits
- Categoriser accuracy (~95%) is on synthetic merchants and is **not better than an exact-match lookup table** here (see `lookup_baseline_accuracy` in `reports/categorizer_metrics.json`). Its value shows on unseen merchant strings with real data.
- Forecast is judged by WAPE vs naive baselines on the last 8 months. On this synthetic data the Random Forest does **not** beat a 3-month moving average for total spend (see `reports/forecast_eval.csv`): with 36 monthly points and mostly random noise there is little to learn. I treat it as a pipeline and evaluation scaffold, and real data would be needed before drawing conclusions.
- Savings numbers are what-ifs on one (synthetic) user's history.
- Optional extension: an LSTM forecaster (TensorFlow); 36 monthly points is too little for it to beat the tree model, so it is left out.

## Results at a glance (simulated data)
![Forecast WAPE](docs/img/forecast_wape.png)

## How this was built
Developed using a combination of GenAI tools and my own coding and implementation. Numbers in this README come from running the code in this repository.

## Author
Pranjal Tomar. Part of my [portfolio](https://github.com/Pranjaltomar07).
