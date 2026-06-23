# tabml

**From a raw CSV to an audited, trained, explained, and served model — in four commands.**

`tabml` is a small, no-config command-line tool for tabular machine learning.
It does the boring-but-easy-to-get-wrong parts correctly: it **audits your data
for target leakage** before you trust a single score, builds a clean
preprocessing pipeline, compares several models with cross-validation, explains
what drove the result, and can **serve the model as a REST API**.

```bash
tabml audit data.csv --target churn     # catch leakage & data issues first
tabml train data.csv --target churn     # compare models, explain, save
tabml serve model.joblib                # instant FastAPI prediction endpoint
tabml predict new.csv -m model.joblib   # batch-score new rows
```

## Why it exists

Most "quick ML" skips the two things that actually bite you:

1. **Target leakage** — a feature that secretly encodes the answer. Your offline
   score looks amazing and then collapses in production. `tabml audit` fits a
   tiny single-feature model per column and flags any feature that predicts the
   target almost perfectly *on its own*:

   ```
   $ tabml audit transactions.csv --target fraud
   ⛔ TARGET LEAKAGE SUSPECTS — investigate before trusting any score:
        settlement_status: predicts 'fraud' almost perfectly alone (AUC 0.998) — likely target leakage
   ⚠ WARN  customer_id: unique per row — looks like an ID; drop before training
   ```
   (`audit` exits non-zero when it finds leakage, so it drops straight into CI.)

2. **Serving** — a trained model is useless in a notebook. `tabml serve` turns a
   saved artifact into a live FastAPI service with `/predict`, `/schema`, and
   `/health` — no glue code.

Everything in between (impute → encode → compare → cross-validate → evaluate →
explain → persist) is done for you, reproducibly.

## Install

```bash
pip install -e .                # core (audit, train, predict)
pip install -e ".[serve]"       # + FastAPI serving
```

Python 3.9+. Core deps: pandas, numpy, scikit-learn, joblib.

## Commands

### `audit` — pre-flight data check
```bash
tabml audit data.csv --target y
```
Flags **target leakage**, ID-like and constant columns, high missingness,
class imbalance, and high-cardinality categoricals.

### `train` — compare, explain, save
```bash
tabml train data.csv --target y --card model_card.md
```
Auto-detects classification vs regression, compares three baselines by
cross-validated `f1_weighted` / `r2`, evaluates the winner on a held-out split,
prints **top feature importances**, and (optionally) writes a markdown
**model card**.

### `serve` — REST API from a model
```bash
tabml serve model.joblib --port 8000
# POST /predict  {"records": [{...}]}  ->  {"predictions": [...]}
```

### `predict` — batch scoring
```bash
tabml predict new.csv --model model.joblib --out scored.csv
```

## Under the hood

- **Leakage scan** — per feature, a depth-limited tree is cross-validated using
  *only that feature*; a near-perfect score (≥ 0.985 AUC/accuracy/R²) is flagged.
- **Preprocessing** (`ColumnTransformer`) — median-impute + standardize numeric;
  most-frequent-impute + one-hot encode categoricals (`handle_unknown="ignore"`).
- **Model comparison** — logistic/linear, random forest, gradient boosting,
  ranked by CV; the winner is refit and scored on a held-out test split.
- **Explainability** — tree importances or linear coefficients mapped back onto
  the expanded feature names.
- **Persistence** — the full fitted pipeline + metadata in one portable
  `joblib` file; `serve`/`predict` need nothing else.

## Library API

```python
from tabml import core, audit

issues = audit.audit(df, target="label")     # -> AuditResult (leaks, warnings)
result = core.train(df, target="label")        # -> TrainResult (metrics, importances)
core.save(result, "model.joblib")
preds = core.predict("model.joblib", new_df)
```

## Develop

```bash
pip install -e ".[dev]"
pytest -q                 # 18 tests (core, audit, serving)
```

## License

[MIT](LICENSE)
