"""Загрузка обученной модели и предсказание вероятности дефолта.

Используется API (``src/api/app.py``), скриптом мониторинга и для пакетного
скоринга из командной строки::

    python -m src.models.predict --input data/processed/test.csv --output reports/test_scores.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import RAW_FEATURES, load_params, resolve_path, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class CreditModel:
    """Обученный Pipeline + порог классификации + метаданные обучения."""

    pipeline: Pipeline
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.get("model_name", type(self.pipeline[-1]).__name__)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Вероятность дефолта (PD) для каждой строки."""
        return self.pipeline.predict_proba(X[RAW_FEATURES])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Класс: 1 — дефолт, 0 — нет (по подобранному порогу)."""
        return (self.predict_proba(X) >= self.threshold).astype(int)


def load_model(
    model_path: str | Path | None = None, meta_path: str | Path | None = None
) -> CreditModel:
    params = load_params()["train"]
    model_path = resolve_path(model_path or params["model_path"])
    meta_path = resolve_path(meta_path or params["meta_path"])

    pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        logger.warning("Model metadata %s not found, using threshold 0.5", meta_path)
    return CreditModel(
        pipeline=pipeline, threshold=float(metadata.get("threshold", 0.5)), metadata=metadata
    )


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Batch scoring with the trained PD model")
    parser.add_argument("--input", required=True, help="CSV with raw client features")
    parser.add_argument("--output", required=True, help="Where to write CSV with scores")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--meta-path", default=None)
    args = parser.parse_args(argv)

    model = load_model(args.model_path, args.meta_path)
    df = pd.read_csv(resolve_path(args.input))
    df["default_probability"] = model.predict_proba(df)
    df["default_prediction"] = (df["default_probability"] >= model.threshold).astype(int)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Scored %d rows with %s -> %s", len(df), model.name, output)


if __name__ == "__main__":
    main()
