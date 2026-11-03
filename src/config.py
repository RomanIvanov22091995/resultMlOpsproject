"""Общие константы проекта и чтение ``params.yaml``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

TARGET = "default.payment.next.month"
ID_COLUMN = "ID"

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Признаки, которые приходят "на вход" модели (сырые поля клиента).
RAW_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_AMT_COLUMNS,
    *PAY_AMT_COLUMNS,
]

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Единый формат логов для всех CLI-скриптов."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # Сторонние библиотеки очень "болтливы" на уровне INFO.
    for noisy in ("great_expectations", "alembic", "mlflow", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Читает YAML с параметрами (по умолчанию ``params.yaml`` в корне)."""
    params_path = Path(path) if path else PARAMS_PATH
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Делает относительный путь абсолютным относительно корня проекта."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
