"""Tests for the model server (skipped if FastAPI isn't installed)."""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from tabml import core

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tabml.serve import build_app  # noqa: E402


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    data = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "label"})
    result = core.train(data, target="label", cv=3)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    core.save(result, path)
    return str(path), data


def test_health_and_schema(model_path):
    path, _ = model_path
    client = TestClient(build_app(path))
    assert client.get("/health").json()["status"] == "ok"
    schema = client.get("/schema").json()
    assert schema["task"] == core.CLASSIFICATION
    assert "features" in schema and len(schema["features"]) > 0


def test_predict_endpoint(model_path):
    path, data = model_path
    client = TestClient(build_app(path))
    record = data.drop(columns=["label"]).iloc[0].to_dict()
    r = client.post("/predict", json={"records": [record]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 1 and "prediction" in preds[0]


def test_missing_features_422(model_path):
    path, _ = model_path
    client = TestClient(build_app(path))
    r = client.post("/predict", json={"records": [{"bogus": 1}]})
    assert r.status_code == 422
