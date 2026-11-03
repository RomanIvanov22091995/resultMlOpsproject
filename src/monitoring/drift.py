"""Метрики дрифта данных: Population Stability Index (PSI).

PSI сравнивает распределение признака (или скора модели) на эталонных данных
(обучающая выборка) и на новых данных::

    PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))

Общепринятая интерпретация в кредитном скоринге:

* PSI < 0.10 — распределение стабильно;
* 0.10 <= PSI < 0.25 — умеренный сдвиг, нужно наблюдать;
* PSI >= 0.25 — значимый сдвиг, модель нужно перепроверить/переобучить.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
# Признак с небольшим числом уникальных значений считаем категориальным.
MAX_CATEGORIES = 15
EPSILON = 1e-4


def psi_status(psi: float) -> str:
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def _psi_from_shares(expected_share: np.ndarray, actual_share: np.ndarray) -> float:
    # Пустые корзины заменяем малым числом, иначе логарифм уходит в бесконечность.
    expected_share = np.clip(expected_share, EPSILON, None)
    actual_share = np.clip(actual_share, EPSILON, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def calculate_psi(expected, actual, bins: int = 10) -> float:
    """PSI для непрерывной величины; корзины — квантили эталонного распределения."""
    expected = pd.Series(expected, dtype=float).dropna().to_numpy()
    actual = pd.Series(actual, dtype=float).dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("PSI requires non-empty expected and actual samples")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    inner_edges = edges[1:-1]
    n_bins = len(inner_edges) + 1
    expected_counts = np.bincount(np.digitize(expected, inner_edges), minlength=n_bins)
    actual_counts = np.bincount(np.digitize(actual, inner_edges), minlength=n_bins)
    return _psi_from_shares(
        expected_counts / expected_counts.sum(), actual_counts / actual_counts.sum()
    )


def calculate_psi_categorical(expected, actual) -> float:
    """PSI для категориального признака: корзины — сами категории."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    categories = sorted(set(expected.unique()) | set(actual.unique()))
    expected_share = expected.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual_share = actual.value_counts(normalize=True).reindex(categories, fill_value=0)
    return _psi_from_shares(expected_share.to_numpy(), actual_share.to_numpy())


def feature_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], bins: int = 10
) -> pd.DataFrame:
    """Таблица PSI по признакам, отсортированная по убыванию сдвига."""
    rows = []
    for feature in features:
        is_categorical = reference[feature].nunique() <= MAX_CATEGORIES
        psi = (
            calculate_psi_categorical(reference[feature], current[feature])
            if is_categorical
            else calculate_psi(reference[feature], current[feature], bins=bins)
        )
        rows.append(
            {
                "feature": feature,
                "type": "categorical" if is_categorical else "numeric",
                "psi": round(psi, 4),
                "status": psi_status(psi),
                "reference_mean": round(float(reference[feature].mean()), 3),
                "current_mean": round(float(current[feature].mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
