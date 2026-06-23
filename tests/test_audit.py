"""Tests for the data audit — leakage detection and quality checks."""

import numpy as np
import pandas as pd

from tabml import audit as A


def _base_df(n=400):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": y})


class TestLeakage:
    def test_detects_a_leaky_feature(self):
        df = _base_df()
        # a feature that *is* the target (with trivial noise) must be flagged
        df["leaky"] = df["label"] + 0.0
        res = A.audit(df, target="label")
        leaked_cols = {f.column for f in res.leaks}
        assert "leaky" in leaked_cols

    def test_clean_features_not_flagged_as_leak(self):
        df = _base_df()
        res = A.audit(df, target="label")
        assert "x1" not in {f.column for f in res.leaks}
        assert "x2" not in {f.column for f in res.leaks}


class TestQualityChecks:
    def test_constant_column_warned(self):
        df = _base_df()
        df["const"] = 7
        res = A.audit(df, target="label")
        assert any(f.column == "const" and "constant" in f.message for f in res.warnings)

    def test_id_like_column_warned(self):
        df = _base_df()
        df["row_id"] = range(len(df))
        res = A.audit(df, target="label")
        msgs = [f.message for f in res.findings if f.column == "row_id"]
        assert any("ID" in m for m in msgs)

    def test_high_missingness_warned(self):
        df = _base_df()
        df.loc[: len(df) // 2, "x1"] = np.nan
        res = A.audit(df, target="label")
        assert any(f.column == "x1" and "missing" in f.message for f in res.warnings)

    def test_imbalance_warned(self):
        rng = np.random.default_rng(1)
        n = 500
        y = (rng.uniform(size=n) < 0.03).astype(int)  # ~3% positive
        df = pd.DataFrame({"x": rng.normal(size=n), "label": y})
        res = A.audit(df, target="label")
        assert any("imbalance" in f.message for f in res.warnings)
