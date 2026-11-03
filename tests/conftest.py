"""Общие фикстуры: выборка данных из репозитория и быстро обученная модель."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from src.api.schemas import EXAMPLE_CLIENT
from src.config import PROJECT_ROOT, RAW_FEATURES, TARGET
from src.data.make_dataset import clean_data
from src.models.pipeline import create_pipeline

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "credit_sample.csv"
TEST_THRESHOLD = 0.3


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """2000 строк исходного датасета (с ID и целевой переменной)."""
    return pd.read_csv(SAMPLE_PATH)


@pytest.fixture(scope="session")
def clean_sample(raw_sample) -> pd.DataFrame:
    return clean_data(raw_sample)


@pytest.fixture(scope="session")
def trained_model_files(tmp_path_factory, clean_sample):
    """Логистическая регрессия, обученная на выборке, + файл метаданных."""
    pipeline = create_pipeline("logistic_regression")
    pipeline.fit(clean_sample[RAW_FEATURES], clean_sample[TARGET])

    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "model_meta.json"
    joblib.dump(pipeline, model_path)
    meta_path.write_text(
        json.dumps({"model_name": "logistic_regression", "threshold": TEST_THRESHOLD}),
        encoding="utf-8",
    )
    return model_path, meta_path


@pytest.fixture
def client_payload() -> dict:
    return dict(EXAMPLE_CLIENT)
