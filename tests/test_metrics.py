import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.models.evaluate import compute_metrics, find_best_threshold, ks_statistic, roc_points

Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])
Y_PROBA = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.8, 0.45, 0.7, 0.05, 0.9])


def test_perfect_model_metrics():
    y_true = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks"] == 1.0
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.0


def test_metrics_match_sklearn():
    metrics = compute_metrics(Y_TRUE, Y_PROBA, threshold=0.5)
    auc = roc_auc_score(Y_TRUE, Y_PROBA)
    assert metrics["roc_auc"] == pytest.approx(auc)
    assert metrics["gini"] == pytest.approx(2 * auc - 1)
    assert metrics["f1"] == pytest.approx(f1_score(Y_TRUE, Y_PROBA >= 0.5))
    expected_keys = {
        "roc_auc",
        "gini",
        "ks",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "brier",
        "log_loss",
    }
    assert set(metrics) == expected_keys


def test_threshold_changes_classification_metrics_only():
    low = compute_metrics(Y_TRUE, Y_PROBA, threshold=0.3)
    high = compute_metrics(Y_TRUE, Y_PROBA, threshold=0.75)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["recall"] > high["recall"]


def test_ks_statistic_bounds():
    assert 0.0 <= ks_statistic(Y_TRUE, Y_PROBA) <= 1.0
    assert ks_statistic([0, 1], [0.9, 0.1]) == 0.0


def test_best_f1_threshold_is_optimal():
    threshold = find_best_threshold(Y_TRUE, Y_PROBA, metric="f1")
    best_f1 = f1_score(Y_TRUE, Y_PROBA >= threshold)
    for candidate in np.unique(Y_PROBA):
        assert best_f1 >= f1_score(Y_TRUE, Y_PROBA >= candidate) - 1e-12


def test_youden_threshold_in_range():
    assert 0.0 <= find_best_threshold(Y_TRUE, Y_PROBA, metric="youden") <= 1.0


def test_unknown_threshold_metric():
    with pytest.raises(ValueError):
        find_best_threshold(Y_TRUE, Y_PROBA, metric="accuracy")


def test_roc_points_monotonic():
    points = roc_points(Y_TRUE, Y_PROBA)
    assert list(points.columns) == ["fpr", "tpr"]
    assert points["fpr"].is_monotonic_increasing
    assert points["tpr"].is_monotonic_increasing
