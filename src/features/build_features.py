"""Feature Engineering для PD-модели.

Все преобразования оформлены как sklearn-трансформер :class:`FeatureEngineer`,
который стоит первым шагом в Pipeline. Благодаря этому модель принимает
"сырые" поля клиента (как в API), а новые признаки считаются одинаково
при обучении и при инференсе — нет расхождения train/serving.

Новые признаки:

* ``AGE_BIN`` — биннинг возраста (категориальный признак);
* агрегаты истории платежей: максимальная/средняя просрочка, число месяцев
  с просрочкой и без использования карты;
* агрегаты сумм: средний счёт, средний платёж;
* ``UTILIZATION_*`` — загрузка кредитного лимита;
* ``PAY_TO_BILL_RATIO`` — какую долю выставленных счетов клиент оплатил;
* ``BILL_GROWTH`` — рост задолженности за полгода относительно лимита.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import BILL_AMT_COLUMNS, PAY_AMT_COLUMNS, PAY_STATUS_COLUMNS, RAW_FEATURES

AGE_BINS = [0, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

ENGINEERED_NUMERIC_FEATURES = [
    "PAY_DELAY_MAX",
    "PAY_DELAY_MEAN",
    "N_MONTHS_DELAYED",
    "N_MONTHS_NO_CONSUMPTION",
    "BILL_AMT_MEAN",
    "PAY_AMT_MEAN",
    "UTILIZATION_LAST",
    "UTILIZATION_MEAN",
    "PAY_TO_BILL_RATIO",
    "BILL_GROWTH",
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "AGE_BIN"]
NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    *PAY_STATUS_COLUMNS[1:],
    *BILL_AMT_COLUMNS,
    *PAY_AMT_COLUMNS,
    *ENGINEERED_NUMERIC_FEATURES,
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def consolidate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Объединяет недокументированные коды категорий в группу "другое".

    EDUCATION: 0, 5, 6 -> 4 (others); MARRIAGE: 0 -> 3 (others).
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float) -> pd.Series:
    result = numerator / denominator.where(denominator > 0)
    return result.fillna(default)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает копию ``df`` с добавленными инженерными признаками."""
    df = consolidate_categories(df)

    df["AGE_BIN"] = pd.cut(df["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(object)

    pay_status = df[PAY_STATUS_COLUMNS]
    delays = pay_status.clip(lower=0)
    df["PAY_DELAY_MAX"] = delays.max(axis=1)
    df["PAY_DELAY_MEAN"] = delays.mean(axis=1)
    df["N_MONTHS_DELAYED"] = (pay_status > 0).sum(axis=1)
    df["N_MONTHS_NO_CONSUMPTION"] = (pay_status == -2).sum(axis=1)

    bills = df[BILL_AMT_COLUMNS]
    payments = df[PAY_AMT_COLUMNS]
    df["BILL_AMT_MEAN"] = bills.mean(axis=1)
    df["PAY_AMT_MEAN"] = payments.mean(axis=1)

    df["UTILIZATION_LAST"] = _safe_divide(df["BILL_AMT1"], df["LIMIT_BAL"], 0.0)
    df["UTILIZATION_MEAN"] = _safe_divide(df["BILL_AMT_MEAN"], df["LIMIT_BAL"], 0.0)

    # Платёж PAY_AMT{k} гасит счёт предыдущего месяца BILL_AMT{k+1}.
    paid = payments[PAY_AMT_COLUMNS[:-1]].sum(axis=1)
    billed = bills[BILL_AMT_COLUMNS[1:]].clip(lower=0).sum(axis=1)
    df["PAY_TO_BILL_RATIO"] = _safe_divide(paid, billed, 1.0).clip(upper=5.0)

    df["BILL_GROWTH"] = _safe_divide(df["BILL_AMT1"] - df["BILL_AMT6"], df["LIMIT_BAL"], 0.0)

    return df.replace([np.inf, -np.inf], np.nan)


def _to_category_strings(series: pd.Series) -> pd.Series:
    """Категории -> строки ('2', '25-34'), пропуски остаются NaN для импьютера."""

    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return str(int(value))
        return str(value)

    return series.map(convert).astype(object)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """sklearn-трансформер: сырые поля клиента -> признаки модели."""

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = build_features(X[RAW_FEATURES])
        out = features[MODEL_FEATURES].copy()
        for column in CATEGORICAL_FEATURES:
            out[column] = _to_category_strings(out[column])
        for column in NUMERIC_FEATURES:
            out[column] = out[column].astype("float64")
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)
