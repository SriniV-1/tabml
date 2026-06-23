"""
Serve a trained model as a REST API. `tabml serve model.joblib` turns a saved
artifact into a live FastAPI service with /predict, /schema and /health — the
"productionize it now" path.

FastAPI/uvicorn are an optional dependency: `pip install tabml[serve]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import core


def build_app(model_path: str | Path):
    """Build a FastAPI app that serves predictions from a saved model."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "serving needs FastAPI — install with: pip install 'tabml[serve]'"
        ) from e

    payload = core.load(model_path)
    features = payload["feature_names"]
    task = payload["task"]

    app = FastAPI(
        title="tabml model server",
        description=f"Serving a {task} model (best: {payload.get('best_model')}).",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return {
            "task": task,
            "target": payload["target"],
            "features": features,
            "best_model": payload.get("best_model"),
            "class_labels": payload.get("class_labels"),
        }

    @app.post("/predict")
    def predict(body: dict) -> dict[str, Any]:
        # Accept {"records": [ {...}, ... ]} or a single {...} record.
        records = body.get("records") if isinstance(body, dict) and "records" in body else body
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records:
            raise HTTPException(status_code=400, detail="send a record or {'records': [...]}")
        df = pd.DataFrame(records)
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"missing features: {missing}")
        out = core.predict(model_path, df)
        cols = [c for c in ("prediction", "confidence") if c in out.columns]
        return {"predictions": out[cols].to_dict(orient="records")}

    return app


def serve(model_path: str | Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "serving needs uvicorn — install with: pip install 'tabml[serve]'"
        ) from e
    uvicorn.run(build_app(model_path), host=host, port=port)
