"""Валидация входящих данных с помощью Great Expectations (GX 1.x).

Набор правил (Expectation Suite) описывает, какими должны быть "нормальные"
данные: все колонки на месте, нет пропусков, типы числовые, значения лежат
в допустимых диапазонах, категориальные коды из известного словаря.
Если в новых данных появляются аномалии, валидация падает и скрипт
завершается с кодом 1 — это останавливает DVC-пайплайн и CI.

Запуск::

    python -m src.data.validation --input data/raw/UCI_Credit_Card.csv
    python -m src.data.validation --input data/sample/credit_sample.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.config import (
    BILL_AMT_COLUMNS,
    ID_COLUMN,
    PAY_AMT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURES,
    TARGET,
    load_params,
    resolve_path,
    setup_logging,
)

logger = logging.getLogger(__name__)

SUITE_NAME = "credit_data_suite"
NUMERIC_TYPES = ["int64", "int32", "float64", "float32", "Int64", "Float64"]

# Допустимые коды категориальных признаков (по описанию датасета + реальные значения).
CATEGORICAL_DOMAINS: dict[str, list[int]] = {
    "SEX": [1, 2],
    "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    "MARRIAGE": [0, 1, 2, 3],
    **{col: list(range(-2, 10)) for col in PAY_STATUS_COLUMNS},
}

# Допустимые диапазоны числовых признаков (бизнес-ограничения, а не min/max выборки).
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "LIMIT_BAL": (1_000, 5_000_000),
    "AGE": (18, 100),
    **{col: (-1_000_000, 10_000_000) for col in BILL_AMT_COLUMNS},
    **{col: (0, 10_000_000) for col in PAY_AMT_COLUMNS},
}

# Доля дефолтов: если она резко изменилась, скорее всего данные испорчены.
TARGET_MEAN_RANGE = (0.05, 0.50)


@dataclass
class ValidationReport:
    """Компактный результат валидации (удобно логировать и сохранять в JSON)."""

    success: bool
    n_rows: int
    evaluated_expectations: int
    successful_expectations: int
    failed_expectations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _get_batch_definition():
    """Эфемерный (in-memory) контекст GX с pandas-источником данных.

    Создаётся один раз на процесс: GX требует активный контекст даже для
    создания ExpectationSuite.
    """
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    data_source = context.data_sources.add_pandas(name="pandas")
    asset = data_source.add_dataframe_asset(name="credit_data")
    return asset.add_batch_definition_whole_dataframe("whole_dataframe")


def build_expectation_suite(
    include_target: bool = True, include_id: bool = False
) -> gx.ExpectationSuite:
    """Создаёт набор правил для данных о клиентах кредитных карт."""
    _get_batch_definition()
    suite = gx.ExpectationSuite(name=SUITE_NAME)
    expectations = gx.expectations
    columns = RAW_FEATURES + ([TARGET] if include_target else [])

    suite.add_expectation(expectations.ExpectTableRowCountToBeBetween(min_value=1))

    if include_id:
        suite.add_expectation(expectations.ExpectColumnToExist(column=ID_COLUMN))
        suite.add_expectation(expectations.ExpectColumnValuesToBeUnique(column=ID_COLUMN))

    for column in columns:
        suite.add_expectation(expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(expectations.ExpectColumnValuesToNotBeNull(column=column))
        suite.add_expectation(
            expectations.ExpectColumnValuesToBeInTypeList(column=column, type_list=NUMERIC_TYPES)
        )

    for column, (min_value, max_value) in NUMERIC_RANGES.items():
        suite.add_expectation(
            expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=min_value, max_value=max_value
            )
        )

    for column, value_set in CATEGORICAL_DOMAINS.items():
        suite.add_expectation(
            expectations.ExpectColumnValuesToBeInSet(column=column, value_set=value_set)
        )

    if include_target:
        suite.add_expectation(
            expectations.ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1])
        )
        suite.add_expectation(
            expectations.ExpectColumnMeanToBeBetween(
                column=TARGET, min_value=TARGET_MEAN_RANGE[0], max_value=TARGET_MEAN_RANGE[1]
            )
        )
    return suite


def suite_to_dict(suite: gx.ExpectationSuite) -> dict[str, Any]:
    """JSON-представление suite без случайных id (стабильный diff в git)."""
    raw = suite.to_json_dict()
    return {
        "name": raw["name"],
        "expectations": [
            {"type": exp["type"], "kwargs": exp["kwargs"]} for exp in raw["expectations"]
        ],
    }


def save_suite(suite: gx.ExpectationSuite, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(suite_to_dict(suite), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def validate_dataframe(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite | None = None,
    include_target: bool = True,
) -> ValidationReport:
    """Проверяет DataFrame набором правил и возвращает компактный отчёт."""
    suite = suite or build_expectation_suite(include_target=include_target)
    batch = _get_batch_definition().get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)

    failed = []
    for item in result.results:
        if item.success:
            continue
        details = item.result or {}
        failed.append(
            {
                "expectation": item.expectation_config.type,
                "kwargs": {
                    k: v for k, v in item.expectation_config.kwargs.items() if k != "batch_id"
                },
                "unexpected_count": details.get("unexpected_count"),
                "observed_value": details.get("observed_value"),
                "examples": details.get("partial_unexpected_list", [])[:5],
            }
        )

    stats = result.statistics
    return ValidationReport(
        success=bool(result.success),
        n_rows=len(df),
        evaluated_expectations=int(stats["evaluated_expectations"]),
        successful_expectations=int(stats["successful_expectations"]),
        failed_expectations=failed,
    )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    params = load_params()
    parser = argparse.ArgumentParser(description="Validate credit data with Great Expectations")
    parser.add_argument("--input", default=params["data"]["raw_path"])
    parser.add_argument("--suite-out", default=params["validation"]["suite_path"])
    parser.add_argument("--report-out", default=params["validation"]["report_path"])
    parser.add_argument(
        "--no-target", action="store_true", help="Data has no target column (scoring data)"
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(resolve_path(args.input))
    suite = build_expectation_suite(
        include_target=not args.no_target, include_id=ID_COLUMN in df.columns
    )
    save_suite(suite, resolve_path(args.suite_out))

    report = validate_dataframe(df, suite)
    report_path = resolve_path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")

    logger.info(
        "Validation of %s: %d/%d expectations passed",
        args.input,
        report.successful_expectations,
        report.evaluated_expectations,
    )
    if not report.success:
        for failure in report.failed_expectations:
            logger.error("FAILED %s %s", failure["expectation"], failure["kwargs"])
        logger.error("Data validation FAILED, see %s", report_path)
        return 1
    logger.info("Data validation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
