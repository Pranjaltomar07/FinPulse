# Architecture

```
statement CSV / simulator
        |  ingest.load_statement   (debit/credit or signed amounts -> one schema)
        v
clean.prepare                       strip channel prefixes, references, city codes; add month
        v
pipeline.categorise                 given labels -> Categoriser -> label + confidence; confidence < 0.6 -> review queue
        |                           (no labels: rules.seed_label first)
        +--> recurring.detect_recurring     timing + amount stability
        +--> anomaly.score                  robust z-score of log amount, per category
        +--> forecast.next_month            ensemble + band from rolling-origin errors
        +--> budget.savings_plan / budget_risk
        v
warehouse.build                     SQLite: transactions, recurring, anomalies + SQL views
        v
dashboard.render                    one self-contained HTML file
```

`evaluate.py` runs the same functions on ten simulated users and scores them against the hidden truth table. No stage reads the truth table.
