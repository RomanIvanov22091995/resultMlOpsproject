"""Сборка sklearn Pipeline и объекта подбора гиперпараметров.

Pipeline = FeatureEngineer -> ColumnTransformer (Imputer + Scaler / Imputer + OneHot)
-> классификатор. Весь препроцессинг живёт внутри модели, поэтому сохранённый
файл модели самодостаточен: на вход ему подаются сырые признаки клиента.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer

CLASSIFIER_STEP = "classifier"


def get_classifier(name: str, random_state: int = 42):
    """Фабрика классификаторов по имени из ``params.yaml``."""
    factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=random_state),
        # n_jobs=1: параллелим на уровне кросс-валидации, а не внутри леса.
        "random_forest": lambda: RandomForestClassifier(random_state=random_state, n_jobs=1),
        "extra_trees": lambda: ExtraTreesClassifier(random_state=random_state, n_jobs=1),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=random_state),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=random_state),
    }
    if name not in factories:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(factories)}")
    return factories[name]()


def create_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features or NUMERIC_FEATURES),
            ("cat", categorical_transformer, categorical_features or CATEGORICAL_FEATURES),
        ]
    )


def create_pipeline(model_name: str = "gradient_boosting", random_state: int = 42) -> Pipeline:
    """Полный пайплайн: инженерия признаков + препроцессинг + модель."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", create_preprocessor()),
            (CLASSIFIER_STEP, get_classifier(model_name, random_state)),
        ]
    )


def to_pipeline_grid(params: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """{'C': [...]} -> {'classifier__C': [...]} — адресация шага внутри Pipeline."""
    return {f"{CLASSIFIER_STEP}__{name}": values for name, values in params.items()}


def build_search(
    pipeline: Pipeline,
    model_config: dict[str, Any],
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: int = 42,
) -> GridSearchCV | RandomizedSearchCV:
    """GridSearchCV (search: grid) или RandomizedSearchCV (search: random)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = to_pipeline_grid(model_config.get("params", {}))
    search_type = model_config.get("search", "grid")

    if search_type == "grid":
        return GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=True)
    if search_type == "random":
        return RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=model_config.get("n_iter", 10),
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
        )
    raise ValueError(f"Unknown search type '{search_type}', expected 'grid' or 'random'")
