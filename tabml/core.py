"""
Core modelling logic: task detection, pipeline construction, training,
evaluation, and persistence. Everything here is plain scikit-learn so a trained
artifact is a single portable joblib file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSIFICATION = "classification"
REGRESSION = "regression"


def detect_task(y: pd.Series, threshold: int = 15) -> str:
    """Heuristically decide classification vs regression from the target.

    Non-numeric targets are always classification. Numeric targets with few
    distinct values (<= threshold) are treated as classification, otherwise
    regression.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return CLASSIFICATION
    nunique = y.nunique(dropna=True)
    if nunique <= threshold and (y.dropna() % 1 == 0).all():
        return CLASSIFICATION
    return REGRESSION


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numeric columns; impute + one-hot encode categoricals."""
    numeric, categorical = _split_columns(X)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def candidate_models(task: str) -> dict[str, Any]:
    if task == CLASSIFICATION:
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=0),
            "gradient_boosting": GradientBoostingClassifier(random_state=0),
        }
    return {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=0),
        "gradient_boosting": GradientBoostingRegressor(random_state=0),
    }


@dataclass
class TrainResult:
    task: str
    target: str
    best_model: str
    cv_scores: dict[str, float]
    cv_metric: str
    test_metrics: dict[str, float]
    n_rows: int
    n_features: int
    feature_names: list[str]
    class_labels: Optional[list] = None
    importances: list = field(default_factory=list)
    elapsed_s: float = 0.0
    pipeline: Any = field(default=None, repr=False)


def train(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 0,
) -> TrainResult:
    """Train and compare candidate models, returning the best fitted pipeline."""
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in data")

    t0 = time.time()
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]
    task = detect_task(y)

    class_labels = None
    if task == CLASSIFICATION:
        class_labels = sorted(y.unique().tolist())

    stratify = y if task == CLASSIFICATION and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    cv_metric = "f1_weighted" if task == CLASSIFICATION else "r2"
    pre = build_preprocessor(X)

    cv_scores: dict[str, float] = {}
    best_name, best_score, best_pipe = None, -np.inf, None
    for name, model in candidate_models(task).items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=cv_metric)
        mean = float(scores.mean())
        cv_scores[name] = round(mean, 4)
        if mean > best_score:
            best_name, best_score, best_pipe = name, mean, pipe

    best_pipe.fit(X_train, y_train)
    test_metrics = evaluate(best_pipe, X_test, y_test, task)

    return TrainResult(
        task=task,
        target=target,
        best_model=best_name,
        cv_scores=cv_scores,
        cv_metric=cv_metric,
        test_metrics=test_metrics,
        n_rows=len(df),
        n_features=X.shape[1],
        feature_names=X.columns.tolist(),
        class_labels=class_labels,
        importances=feature_importances(best_pipe),
        elapsed_s=round(time.time() - t0, 2),
        pipeline=best_pipe,
    )


def feature_importances(pipe: Pipeline, top: int = 15) -> list[tuple[str, float]]:
    """Top normalized feature importances from the fitted pipeline.

    Uses tree ``feature_importances_`` or linear ``coef_`` magnitude, mapped
    back onto the expanded (post-encoding) feature names.
    """
    pre = pipe.named_steps.get("pre")
    model = pipe.named_steps.get("model")
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        return []
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = np.abs(np.asarray(model.feature_importances_, dtype=float))
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if imp is None or len(imp) != len(names):
        return []
    total = float(imp.sum()) or 1.0
    order = np.argsort(imp)[::-1][:top]
    return [(names[i], round(float(imp[i] / total), 4)) for i in order]


def evaluate(pipe: Pipeline, X, y, task: str) -> dict[str, float]:
    preds = pipe.predict(X)
    if task == CLASSIFICATION:
        metrics = {
            "accuracy": round(accuracy_score(y, preds), 4),
            "f1_weighted": round(f1_score(y, preds, average="weighted"), 4),
        }
        # ROC-AUC only for binary targets with probability support
        if len(np.unique(y)) == 2 and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X)[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y, proba), 4)
            except Exception:
                pass
        return metrics
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return {
        "r2": round(r2_score(y, preds), 4),
        "mae": round(float(mean_absolute_error(y, preds)), 4),
        "rmse": round(rmse, 4),
    }


def save(result: TrainResult, path: str | Path) -> None:
    import joblib

    path = Path(path)
    payload = {
        "pipeline": result.pipeline,
        "task": result.task,
        "target": result.target,
        "feature_names": result.feature_names,
        "class_labels": result.class_labels,
        "best_model": result.best_model,
        "version": 1,
    }
    joblib.dump(payload, path)


def load(path: str | Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def predict(model_path: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a saved model to new rows, returning predictions (+ probabilities)."""
    payload = load(model_path)
    pipe = payload["pipeline"]
    features = payload["feature_names"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required feature columns: {missing}")
    X = df[features]
    out = df.copy()
    out["prediction"] = pipe.predict(X)
    if payload["task"] == CLASSIFICATION and hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            labels = payload.get("class_labels") or list(range(proba.shape[1]))
            out["confidence"] = proba.max(axis=1).round(4)
            _ = labels  # labels available for callers that want per-class columns
        except Exception:
            pass
    return out
