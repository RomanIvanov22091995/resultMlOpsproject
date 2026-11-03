import pandas as pd
import pytest

from src.config import ID_COLUMN, RAW_FEATURES, TARGET
from src.data.make_dataset import clean_data, load_raw_data, split_data


def test_clean_data_drops_id_and_duplicates(raw_sample):
    df = raw_sample.head(100)
    duplicate = df.iloc[[0]].copy()
    duplicate[ID_COLUMN] = 999_999  # тот же клиент с другим ID
    dirty = pd.concat([df, duplicate], ignore_index=True)

    cleaned = clean_data(dirty)

    assert ID_COLUMN not in cleaned.columns
    assert len(cleaned) == 100
    assert list(cleaned.columns) == RAW_FEATURES + [TARGET]


def test_clean_data_drops_missing_target_and_casts_types(raw_sample):
    df = raw_sample.head(10).copy()
    df[TARGET] = df[TARGET].astype("float64")
    df.loc[0, TARGET] = None

    cleaned = clean_data(df)

    assert len(cleaned) == 9
    assert cleaned[TARGET].dtype == "int64"
    assert cleaned["AGE"].dtype == "int64"
    assert cleaned["LIMIT_BAL"].dtype == "float64"


def test_split_is_stratified(clean_sample):
    train, test = split_data(clean_sample, test_size=0.2, random_state=0)

    assert len(train) + len(test) == len(clean_sample)
    assert len(test) == pytest.approx(0.2 * len(clean_sample), abs=1)
    assert train[TARGET].mean() == pytest.approx(test[TARGET].mean(), abs=0.01)


def test_split_is_reproducible(clean_sample):
    first, _ = split_data(clean_sample, random_state=42)
    second, _ = split_data(clean_sample, random_state=42)
    pd.testing.assert_frame_equal(first, second)


def test_load_raw_data_requires_all_columns(tmp_path, raw_sample):
    path = tmp_path / "broken.csv"
    raw_sample.drop(columns=["AGE"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="AGE"):
        load_raw_data(path)
