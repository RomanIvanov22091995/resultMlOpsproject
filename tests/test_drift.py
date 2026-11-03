import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    calculate_psi,
    calculate_psi_categorical,
    feature_drift_report,
    psi_status,
)
from src.monitoring.simulate_traffic import apply_drift

rng = np.random.default_rng(0)
REFERENCE = rng.normal(0, 1, 20_000)


def test_psi_identical_distribution_is_near_zero():
    assert calculate_psi(REFERENCE, rng.normal(0, 1, 5_000)) < 0.01


def test_psi_detects_shift():
    assert calculate_psi(REFERENCE, rng.normal(1.0, 1, 5_000)) > 0.25


def test_psi_grows_with_shift():
    small = calculate_psi(REFERENCE, rng.normal(0.2, 1, 5_000))
    large = calculate_psi(REFERENCE, rng.normal(0.8, 1, 5_000))
    assert 0 < small < large


def test_psi_handles_empty_bins_and_ties():
    expected = np.array([0] * 900 + list(range(1, 101)))  # много одинаковых нулей
    actual = np.full(500, 1000.0)  # все значения вне эталонного диапазона
    psi = calculate_psi(expected, actual)
    assert np.isfinite(psi)
    assert psi > 0.25


def test_psi_requires_data():
    with pytest.raises(ValueError):
        calculate_psi([], [1, 2, 3])


def test_categorical_psi():
    expected = ["a"] * 50 + ["b"] * 50
    assert calculate_psi_categorical(expected, ["a"] * 50 + ["b"] * 50) == pytest.approx(0.0)
    assert calculate_psi_categorical(expected, ["a"] * 10 + ["c"] * 90) > 0.25


@pytest.mark.parametrize(
    "psi, status", [(0.0, "stable"), (0.099, "stable"), (0.1, "moderate"), (0.3, "significant")]
)
def test_psi_status(psi, status):
    assert psi_status(psi) == status


def test_feature_drift_report(clean_sample):
    reference = clean_sample.iloc[:1000]
    current = apply_drift(clean_sample.iloc[1000:].reset_index(drop=True))

    report = feature_drift_report(reference, current, ["LIMIT_BAL", "AGE", "PAY_0", "SEX"])

    assert list(report.columns) == [
        "feature",
        "type",
        "psi",
        "status",
        "reference_mean",
        "current_mean",
    ]
    assert report["psi"].is_monotonic_decreasing
    statuses = dict(zip(report["feature"], report["status"]))
    assert statuses["LIMIT_BAL"] == "significant"
    assert statuses["SEX"] == "stable"
    assert dict(zip(report["feature"], report["type"]))["PAY_0"] == "categorical"


def test_apply_drift_keeps_values_valid(clean_sample):
    drifted = apply_drift(clean_sample)
    assert drifted["AGE"].max() <= 100
    assert drifted["PAY_0"].max() <= 9
    assert drifted["LIMIT_BAL"].mean() < clean_sample["LIMIT_BAL"].mean()
    pd.testing.assert_series_equal(drifted["SEX"], clean_sample["SEX"])
