import numpy as np
import pandas as pd
import pytest

from src.api.schemas import EXAMPLE_CLIENT
from src.config import RAW_FEATURES
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    FeatureEngineer,
    build_features,
    consolidate_categories,
)


def make_client(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**EXAMPLE_CLIENT, **overrides}])


def test_consolidate_categories_merges_undocumented_codes():
    df = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6], "MARRIAGE": [0, 1, 2, 3, 1, 2, 3]})
    result = consolidate_categories(df)
    assert result["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]
    assert result["MARRIAGE"].tolist() == [3, 1, 2, 3, 1, 2, 3]


def test_payment_history_aggregates():
    df = make_client(PAY_0=2, PAY_2=3, PAY_3=0, PAY_4=-1, PAY_5=-2, PAY_6=-2)
    row = build_features(df).iloc[0]
    assert row["PAY_DELAY_MAX"] == 3
    assert row["PAY_DELAY_MEAN"] == pytest.approx(5 / 6)
    assert row["N_MONTHS_DELAYED"] == 2
    assert row["N_MONTHS_NO_CONSUMPTION"] == 2


def test_amount_ratios():
    df = make_client(
        LIMIT_BAL=100_000,
        BILL_AMT1=50_000,
        BILL_AMT6=10_000,
        **{f"BILL_AMT{i}": 20_000 for i in range(2, 6)},
        **{f"PAY_AMT{i}": 5_000 for i in range(1, 7)},
    )
    row = build_features(df).iloc[0]
    assert row["UTILIZATION_LAST"] == pytest.approx(0.5)
    assert row["BILL_GROWTH"] == pytest.approx(0.4)
    # оплачено 5 * 5000 = 25000 из выставленных 4 * 20000 + 10000 = 90000
    assert row["PAY_TO_BILL_RATIO"] == pytest.approx(25_000 / 90_000)


def test_no_bills_means_fully_paid_and_no_infinities():
    df = make_client(**{f"BILL_AMT{i}": 0 for i in range(1, 7)}, LIMIT_BAL=0)
    features = build_features(df)
    row = features.iloc[0]
    assert row["PAY_TO_BILL_RATIO"] == 1.0
    assert row["UTILIZATION_LAST"] == 0.0
    assert not np.isinf(features.select_dtypes("number").to_numpy()).any()


@pytest.mark.parametrize(
    "age, expected_bin",
    [(21, "<25"), (25, "25-34"), (34, "25-34"), (35, "35-44"), (64, "55-64"), (79, "65+")],
)
def test_age_binning(age, expected_bin):
    assert build_features(make_client(AGE=age)).loc[0, "AGE_BIN"] == expected_bin


def test_feature_engineer_output_schema(clean_sample):
    transformer = FeatureEngineer().fit(clean_sample)
    out = transformer.transform(clean_sample)

    assert list(out.columns) == MODEL_FEATURES
    assert len(out) == len(clean_sample)
    assert not out.isna().any().any()
    for column in CATEGORICAL_FEATURES:
        assert out[column].map(type).eq(str).all(), column
    for column in NUMERIC_FEATURES:
        assert out[column].dtype == "float64", column
    assert list(transformer.get_feature_names_out()) == MODEL_FEATURES


def test_feature_engineer_ignores_extra_columns_and_order(clean_sample):
    shuffled = clean_sample[RAW_FEATURES[::-1] + ["default.payment.next.month"]].head(5)
    out = FeatureEngineer().fit(shuffled).transform(shuffled)
    assert list(out.columns) == MODEL_FEATURES


def test_feature_engineer_requires_raw_columns(clean_sample):
    with pytest.raises(ValueError, match="LIMIT_BAL"):
        FeatureEngineer().fit(clean_sample.drop(columns=["LIMIT_BAL"]))
