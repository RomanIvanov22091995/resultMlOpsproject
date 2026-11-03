"""REST API для PD-модели на FastAPI.

Запуск локально::

    uvicorn src.api.app:app --reload --port 8000

Документация (Swagger UI): http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src import __version__
from src.api.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    ClientData,
    HealthResponse,
    PredictionResponse,
)
from src.config import setup_logging
from src.models.predict import CreditModel, load_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель один раз при старте сервиса, а не на каждый запрос."""
    setup_logging()
    model_path = os.getenv("MODEL_PATH")
    meta_path = os.getenv("MODEL_META_PATH")
    try:
        app.state.model = load_model(model_path, meta_path)
        logger.info(
            "Loaded model '%s' (threshold %.4f)", app.state.model.name, app.state.model.threshold
        )
    except FileNotFoundError as exc:
        # Сервис стартует, но /health честно сообщает, что модели нет.
        logger.error("Model is not available: %s", exc)
        app.state.model = None
    yield


app = FastAPI(
    title="Credit Default Prediction API",
    description="Predicts the probability of credit card default (PD model).",
    version=__version__,
    lifespan=lifespan,
)


def _get_model(request: Request) -> CreditModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def _score(model: CreditModel, clients: list[ClientData]) -> list[PredictionResponse]:
    frame = pd.DataFrame([client.model_dump() for client in clients])
    probabilities = model.predict_proba(frame)
    return [
        PredictionResponse(
            default_prediction=int(probability >= model.threshold),
            default_probability=round(float(probability), 6),
            threshold=round(model.threshold, 6),
            model_name=model.name,
        )
        for probability in probabilities
    ]


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Credit Default Prediction API is alive!", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded = getattr(request.app.state, "model", None) is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return HealthResponse(status="ok", model_loaded=True)


@app.get("/model-info")
def model_info(request: Request) -> dict:
    """Метаданные обученной модели: алгоритм, порог, метрики, версия в MLflow."""
    model = _get_model(request)
    keys = (
        "model_name",
        "threshold",
        "test_metrics",
        "best_params",
        "trained_at",
        "mlflow_run_id",
        "registered_model_version",
    )
    return {key: model.metadata.get(key) for key in keys}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData, request: Request):
    """Предсказание для одного клиента: класс (0/1) и вероятность дефолта."""
    return _score(_get_model(request), [client])[0]


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchRequest, request: Request):
    """Пакетное предсказание для списка клиентов."""
    return BatchPredictionResponse(predictions=_score(_get_model(request), batch.clients))
