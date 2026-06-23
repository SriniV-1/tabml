"""Tests for the tabml core — task detection, training, evaluation, round-trip."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from tabml import core


def _classification_df():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "label"})
    return df, "label"


def _regression_df():
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "y"})
    return df, "y"


class TestTaskDetection:
    def test_numeric_few_classes_is_classification(self):
        assert core.detect_task(pd.Series([0, 1, 1, 0, 1])) == core.CLASSIFICATION

    def test_string_target_is_classification(self):
        assert core.detect_task(pd.Series(["a", "b", "a"])) == core.CLASSIFICATION

    def test_continuous_is_regression(self):
        assert core.detect_task(pd.Series(np.linspace(0, 100, 200))) == core.REGRESSION


class TestTrainClassification:
    def setup_method(self):
        self.df, self.target = _classification_df()

    def test_detects_classification_and_scores_well(self):
        r = core.train(self.df, target=self.target, cv=3)
        assert r.task == core.CLASSIFICATION
        assert r.class_labels == [0, 1]
        # breast cancer is an easy dataset; any sane pipeline clears 0.9 F1
        assert r.test_metrics["f1_weighted"] > 0.9
        assert "roc_auc" in r.test_metrics  # binary target

    def test_best_model_is_among_candidates(self):
        r = core.train(self.df, target=self.target, cv=3)
        assert r.best_model in core.candidate_models(core.CLASSIFICATION)


class TestTrainRegression:
    def test_detects_regression_and_reports_metrics(self):
        df, target = _regression_df()
        r = core.train(df, target=target, cv=3)
        assert r.task == core.REGRESSION
        assert set(r.test_metrics) == {"r2", "mae", "rmse"}


class TestPersistenceRoundTrip:
    def test_save_load_predict(self, tmp_path):
        df, target = _classification_df()
        r = core.train(df, target=target, cv=3)
        model_path = tmp_path / "m.joblib"
        core.save(r, model_path)

        new_rows = df.drop(columns=[target]).head(5)
        out = core.predict(model_path, new_rows)
        assert "prediction" in out.columns
        assert "confidence" in out.columns
        assert len(out) == 5


class TestErrors:
    def test_missing_target_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            core.train(df, target="nope")

    def test_predict_missing_features_raises(self, tmp_path):
        df, target = _classification_df()
        r = core.train(df, target=target, cv=3)
        path = tmp_path / "m.joblib"
        core.save(r, path)
        with pytest.raises(ValueError):
            core.predict(path, pd.DataFrame({"unrelated": [1, 2]}))
