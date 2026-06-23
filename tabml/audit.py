"""
Pre-flight data audit. Run *before* training to catch the problems that quietly
ruin tabular models — most importantly **target leakage**, where a feature
encodes the answer and inflates offline scores that collapse in production.

The headline check fits a tiny single-feature model per column: if one feature
alone predicts the target almost perfectly, it is flagged as a leakage suspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .core import CLASSIFICATION, detect_task

LEAK_THRESHOLD = 0.985   # single-feature CV score above this => leakage suspect
MAX_LEAK_FEATURES = 120  # skip the per-feature scan beyond this width


@dataclass
class Finding:
    severity: str   # "leak" | "warn" | "info"
    column: str
    message: str


@dataclass
class AuditResult:
    task: str
    target: str
    n_rows: int
    n_features: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def leaks(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "leak"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]


def _encode_single(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        return col.fillna(col.median()).to_numpy().reshape(-1, 1)
    codes = col.astype("category").cat.codes  # -1 for NaN
    return codes.to_numpy().reshape(-1, 1)


def _single_feature_score(x: np.ndarray, y: pd.Series, task: str) -> float | None:
    """Cross-validated score of a shallow tree using only this one feature."""
    try:
        if task == CLASSIFICATION:
            model = DecisionTreeClassifier(max_depth=4, random_state=0)
            scoring = "roc_auc" if y.nunique() == 2 else "accuracy"
        else:
            model = DecisionTreeRegressor(max_depth=4, random_state=0)
            scoring = "r2"
        scores = cross_val_score(model, x, y, cv=3, scoring=scoring)
        return float(scores.mean())
    except Exception:
        return None


def audit(df: pd.DataFrame, target: str) -> AuditResult:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in data")

    df = df.dropna(subset=[target])
    y = df[target]
    task = detect_task(y)
    features = [c for c in df.columns if c != target]
    res = AuditResult(task=task, target=target, n_rows=len(df), n_features=len(features))

    # --- class balance ---
    if task == CLASSIFICATION:
        counts = y.value_counts(normalize=True)
        minority = counts.min()
        if minority < 0.10:
            res.findings.append(Finding(
                "warn", target,
                f"class imbalance: smallest class is {minority:.1%} of rows"))

    for col in features:
        s = df[col]
        nunique = s.nunique(dropna=True)
        missing = s.isna().mean()

        # constant / id-like / high-missing / high-cardinality
        if nunique <= 1:
            res.findings.append(Finding("warn", col, "constant column (no signal)"))
            continue
        if nunique == len(df):
            res.findings.append(Finding(
                "warn", col, "unique per row — looks like an ID; drop before training"))
        if missing > 0.20:
            res.findings.append(Finding("warn", col, f"{missing:.0%} missing values"))
        if (not pd.api.types.is_numeric_dtype(s)) and nunique > 50:
            res.findings.append(Finding(
                "info", col, f"high-cardinality categorical ({nunique} levels)"))

    # --- leakage scan (the headline) ---
    if len(features) <= MAX_LEAK_FEATURES:
        for col in features:
            if df[col].nunique(dropna=True) <= 1:
                continue
            score = _single_feature_score(_encode_single(df[col]), y, task)
            if score is not None and score >= LEAK_THRESHOLD:
                metric = ("AUC" if (task == CLASSIFICATION and y.nunique() == 2)
                          else "accuracy" if task == CLASSIFICATION else "R²")
                res.findings.append(Finding(
                    "leak", col,
                    f"predicts '{target}' almost perfectly alone "
                    f"({metric} {score:.3f}) — likely target leakage"))
    else:
        res.findings.append(Finding(
            "info", "*", f"{len(features)} features — leakage scan skipped (too wide)"))

    return res
