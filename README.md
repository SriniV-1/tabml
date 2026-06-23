# tabml

**Train, evaluate, and serve a model from any CSV — in one command.**

`tabml` is a small, no-config command-line tool for tabular machine learning.
Point it at a CSV and a target column; it detects whether the problem is
classification or regression, builds a clean preprocessing pipeline, compares
several models with cross-validation, prints a readable report, and saves a
portable model you can predict with later.

```bash
$ tabml train passengers.csv --target survived
────────────────────────────────────────────────────────
  tabml · CLASSIFICATION  —  target: survived
────────────────────────────────────────────────────────
  rows: 300   features: 4   trained in 0.7s

  Model comparison (f1_weighted, 3 models, CV):
   ► logistic_regression   0.7030  ████████████████
     gradient_boosting     0.6282  ███████████████
     random_forest         0.6277  ███████████████

  Best model: logistic_regression
  Held-out test metrics:
    accuracy       0.55
    f1_weighted    0.5534
    roc_auc        0.5891
────────────────────────────────────────────────────────
  ✓ model saved → model.joblib
```

## Why

Most "quick ML" still means boilerplate: read the CSV, guess the task, impute,
scale, encode, split, try a few models, score them, pick one, persist it.
`tabml` does exactly that — correctly and reproducibly — so a baseline is one
command instead of a notebook.

## Install

```bash
pip install -e .          # from a clone
# or
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: pandas, numpy, scikit-learn, joblib.

## Usage

**Train** — auto-detects classification vs regression from the target:

```bash
tabml train data.csv --target price            # regression
tabml train data.csv --target churn -o churn.joblib --cv 10
```

**Predict** — apply a saved model to new rows:

```bash
tabml predict new.csv --model churn.joblib --out scored.csv
```

For classification, predictions include a `confidence` column (max class
probability); for regression, a numeric `prediction`.

## What it does under the hood

1. **Task detection** — non-numeric or low-cardinality integer targets →
   classification; continuous targets → regression.
2. **Preprocessing** (`ColumnTransformer`): median-impute + standardize numeric
   columns; most-frequent-impute + one-hot encode categoricals
   (`handle_unknown="ignore"`, so unseen categories at predict time are safe).
3. **Model comparison** — three sensible baselines per task
   (linear/logistic, random forest, gradient boosting), ranked by
   cross-validated `f1_weighted` (classification) or `r2` (regression).
4. **Evaluation** — the winner is refit and scored on a held-out test split:
   accuracy / weighted-F1 / ROC-AUC, or R² / MAE / RMSE.
5. **Persistence** — the full fitted pipeline plus metadata is saved as a single
   `joblib` artifact, so `predict` needs nothing but the file.

## Library API

```python
from tabml import core
import pandas as pd

df = pd.read_csv("data.csv")
result = core.train(df, target="label")
print(result.best_model, result.test_metrics)
core.save(result, "model.joblib")

preds = core.predict("model.joblib", df.drop(columns=["label"]))
```

## Develop

```bash
pip install -e ".[dev]"
pytest -q                 # 9 tests
```

## License

[MIT](LICENSE)
