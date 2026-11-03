"""Тесты валидации данных (Great Expectations).

``test_*_passes_validation`` упадут, если в данных появятся аномалии,
а параметризованный тест проверяет, что каждый тип аномалии действительно
ловится набором правил.
"""

import json

import numpy as np
import pytest

from src.config import PROJECT_ROOT, TARGET
from src.data.validation import build_expectation_suite, main, save_suite, validate_dataframe

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"


def test_sample_data_passes_validation(clean_sample):
    report = validate_dataframe(clean_sample)
    assert report.success, report.failed_expectations
    assert report.evaluated_expectations == report.successful_expectations


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="full dataset is pulled with DVC")
def test_full_raw_data_passes_validation(tmp_path):
    args = [
        "--input",
        str(RAW_DATA_PATH),
        "--suite-out",
        str(tmp_path / "suite.json"),
        "--report-out",
        str(tmp_path / "report.json"),
    ]
    assert main(args) == 0


def _set_nan(df):
    df["LIMIT_BAL"] = df["LIMIT_BAL"].astype(float)
    df.loc[0, "LIMIT_BAL"] = np.nan


def _set_age(df):
    df.loc[0, "AGE"] = 150


def _set_education(df):
    df.loc[0, "EDUCATION"] = 9


def _set_target(df):
    df.loc[0, TARGET] = 2


def _drop_column(df):
    df.drop(columns=["PAY_0"], inplace=True)


def _negative_payment(df):
    df.loc[0, "PAY_AMT1"] = -100.0


def _string_type(df):
    df["AGE"] = df["AGE"].astype(str)


def _all_defaults(df):
    df[TARGET] = 1


@pytest.mark.parametrize(
    "corrupt, expected_failure",
    [
        (_set_nan, "expect_column_values_to_not_be_null"),
        (_set_age, "expect_column_values_to_be_between"),
        (_set_education, "expect_column_values_to_be_in_set"),
        (_set_target, "expect_column_values_to_be_in_set"),
        (_drop_column, "expect_column_to_exist"),
        (_negative_payment, "expect_column_values_to_be_between"),
        (_string_type, "expect_column_values_to_be_in_type_list"),
        (_all_defaults, "expect_column_mean_to_be_between"),
    ],
    ids=[
        "null",
        "age_out_of_range",
        "unknown_category",
        "bad_target",
        "missing_column",
        "negative_payment",
        "wrong_type",
        "target_distribution",
    ],
)
def test_validation_detects_anomalies(clean_sample, corrupt, expected_failure):
    df = clean_sample.head(300).copy()
    corrupt(df)

    report = validate_dataframe(df)

    assert not report.success
    assert expected_failure in {f["expectation"] for f in report.failed_expectations}


def test_validation_without_target(clean_sample):
    report = validate_dataframe(clean_sample.drop(columns=[TARGET]), include_target=False)
    assert report.success


def test_cli_returns_error_code_on_anomalies(tmp_path, clean_sample):
    bad = clean_sample.head(100).copy()
    bad.loc[0, "AGE"] = 5
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    report_path = tmp_path / "report.json"

    code = main(
        [
            "--input",
            str(path),
            "--suite-out",
            str(tmp_path / "suite.json"),
            "--report-out",
            str(report_path),
        ]
    )

    assert code == 1
    assert json.loads(report_path.read_text())["success"] is False


def test_suite_is_saved_without_random_ids(tmp_path):
    path = save_suite(build_expectation_suite(), tmp_path / "suite.json")
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["name"] == "credit_data_suite"
    assert len(content["expectations"]) > 50
    assert all(set(exp) == {"type", "kwargs"} for exp in content["expectations"])
