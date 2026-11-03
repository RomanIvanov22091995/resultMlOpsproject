"""Загрузка, первичная очистка и разбиение данных на train/test.

Запуск (стадия ``prepare`` DVC-пайплайна)::

    python -m src.data.make_dataset
    python -m src.data.make_dataset --input data/raw/UCI_Credit_Card.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BILL_AMT_COLUMNS,
    ID_COLUMN,
    PAY_AMT_COLUMNS,
    RAW_FEATURES,
    TARGET,
    load_params,
    resolve_path,
    setup_logging,
)

logger = logging.getLogger(__name__)

INT_COLUMNS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]
FLOAT_COLUMNS = ["LIMIT_BAL", *BILL_AMT_COLUMNS, *PAY_AMT_COLUMNS]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Читает исходный CSV и проверяет, что в нём есть все нужные колонки."""
    df = pd.read_csv(path)
    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    logger.info("Loaded raw data %s: %d rows x %d columns", path, *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Первичная очистка.

    * удаляет технический идентификатор ``ID`` (не несёт информации о риске);
    * удаляет строки без целевой переменной;
    * удаляет полные дубликаты записей (после удаления ``ID``);
    * приводит типы: категориальные/счётные поля -> int, суммы -> float.

    Перекодировка недокументированных категорий и создание новых признаков
    выполняются в :mod:`src.features.build_features` внутри sklearn Pipeline,
    чтобы одни и те же преобразования применялись и при обучении, и в API.
    """
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d rows (missing target / duplicates)", n_before - len(df))

    df[INT_COLUMNS] = df[INT_COLUMNS].astype("int64")
    df[FLOAT_COLUMNS] = df[FLOAT_COLUMNS].astype("float64")
    df[TARGET] = df[TARGET].astype("int64")
    return df[RAW_FEATURES + [TARGET]]


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение: доля дефолтов в train и test одинакова."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )
    logger.info(
        "Split: train=%d (default rate %.3f), test=%d (default rate %.3f)",
        len(train_df),
        train_df[TARGET].mean(),
        len(test_df),
        test_df[TARGET].mean(),
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    params = load_params()
    parser = argparse.ArgumentParser(description="Prepare train/test datasets")
    parser.add_argument("--input", default=params["data"]["raw_path"])
    parser.add_argument("--output-dir", default=params["data"]["processed_dir"])
    parser.add_argument("--test-size", type=float, default=params["data"]["test_size"])
    parser.add_argument("--random-state", type=int, default=params["base"]["random_state"])
    args = parser.parse_args(argv)

    df = clean_data(load_raw_data(resolve_path(args.input)))
    train_df, test_df = split_data(df, args.test_size, args.random_state)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    logger.info("Saved train.csv and test.csv to %s", output_dir)


if __name__ == "__main__":
    main()
