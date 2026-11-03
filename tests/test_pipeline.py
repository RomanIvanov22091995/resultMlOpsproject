import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import RAW_FEATURES, TARGET, load_params
from src.models.pipeline import build_search, create_pipeline, get_classifier, to_pipeline_grid
from src.models.predict import CreditModel, load_model

MODEL_NAMES = list(load_params()["models"])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_configured_model_trains_and_predicts(model_name, clean_sample):
    data = clean_sample.head(600)
    pipeline = create_pipeline(model_name).fit(data[RAW_FEATURES], data[TARGET])

    proba = pipeline.predict_proba(data[RAW_FEATURES].head(20))[:, 1]

    assert proba.shape == (20,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_pipeline_steps():
    pipeline = create_pipeline("logistic_regression")
    assert list(pipeline.named_steps) == ["features", "preprocessor", "classifier"]


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_classifier("xgboost_super")


def test_param_grid_is_prefixed():
    assert to_pipeline_grid({"C": [1, 2]}) == {"classifier__C": [1, 2]}


def test_build_search_types():
    pipeline = create_pipeline("logistic_regression")
    grid = build_search(pipeline, {"search": "grid", "params": {"C": [0.1, 1]}}, cv_folds=3)
    random = build_search(
        pipeline, {"search": "random", "n_iter": 2, "params": {"C": [0.1, 1, 10]}}, cv_folds=3
    )
    assert isinstance(grid, GridSearchCV)
    assert isinstance(random, RandomizedSearchCV)
    assert random.n_iter == 2
    with pytest.raises(ValueError):
        build_search(pipeline, {"search": "bayes", "params": {}})


def test_search_fits_and_selects_best(clean_sample):
    data = clean_sample.head(500)
    search = build_search(
        create_pipeline("logistic_regression"),
        {"search": "grid", "params": {"C": [0.01, 1.0]}},
        cv_folds=3,
        n_jobs=1,
    )
    search.fit(data[RAW_FEATURES], data[TARGET])
    assert search.best_params_["classifier__C"] in (0.01, 1.0)
    assert 0.5 < search.best_score_ <= 1.0


def test_credit_model_applies_threshold(trained_model_files, clean_sample):
    model = load_model(*trained_model_files)
    assert isinstance(model, CreditModel)
    assert model.threshold == 0.3
    assert model.name == "logistic_regression"

    X = clean_sample.head(50)
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(model.predict(X), (proba >= 0.3).astype(int))
