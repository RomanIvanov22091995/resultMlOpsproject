"""Обучение PD-модели: подбор гиперпараметров + MLflow Tracking.

Для каждого алгоритма из ``params.yaml`` (секция ``models``):

1. GridSearchCV / RandomizedSearchCV по стратифицированной кросс-валидации;
2. каждая комбинация гиперпараметров логируется во вложенный (nested) run;
3. порог классификации подбирается по out-of-fold предсказаниям на train;
4. на отложенной тестовой выборке считаются ROC-AUC, Gini, KS, Precision,
   Recall, F1 и др.; строятся ROC/PR-кривые, матрица ошибок, калибровка;
5. модель логируется в MLflow (``mlflow.sklearn.log_model``).

Лучшая модель выбирается по CV ROC-AUC (не по тесту — чтобы тест оставался
честной оценкой), сохраняется в ``models/`` и регистрируется в MLflow
Model Registry с алиасом ``champion``.

Запуск (стадия ``train`` DVC-пайплайна)::

    python -m src.models.train
    python -m src.models.train --models logistic_regression --cv-folds 3   # быстрый прогон
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import skops
import skops.io as sio
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, RAW_FEATURES, TARGET, load_params, resolve_path, setup_logging
from src.models.evaluate import (
    compute_metrics,
    find_best_threshold,
    roc_points,
    save_evaluation_plots,
)
from src.models.pipeline import CLASSIFIER_STEP, build_search, create_pipeline

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_path: Path
    test_path: Path


@dataclass
class ExperimentResult:
    model_name: str
    run_id: str
    model_uri: str
    pipeline: Pipeline
    cv_score: float
    cv_std: float
    threshold: float
    best_params: dict[str, Any]
    test_metrics: dict[str, float]
    test_metrics_default: dict[str, float]
    fit_time_sec: float


def load_dataset(processed_dir: str | Path) -> Dataset:
    processed_dir = resolve_path(processed_dir)
    train_path, test_path = processed_dir / "train.csv", processed_dir / "test.csv"
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    return Dataset(
        X_train=train_df[RAW_FEATURES],
        y_train=train_df[TARGET],
        X_test=test_df[RAW_FEATURES],
        y_test=test_df[TARGET],
        train_path=train_path,
        test_path=test_path,
    )


def setup_mlflow(mlflow_config: dict[str, Any]) -> None:
    """Подключается к трекинг-серверу (по умолчанию локальная SQLite-БД)."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", mlflow_config["tracking_uri"]))
    experiment_name = mlflow_config["experiment_name"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_location = resolve_path(mlflow_config.get("artifact_location", "mlruns"))
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location.as_uri())
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow tracking URI: %s, experiment: %s", mlflow.get_tracking_uri(), experiment_name
    )


def _strip_prefix(params: dict[str, Any]) -> dict[str, Any]:
    return {key.replace(f"{CLASSIFIER_STEP}__", ""): value for key, value in params.items()}


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{name}": value for name, value in metrics.items()}


def log_search_candidates(search, model_name: str, scoring: str) -> None:
    """Каждая проверенная комбинация гиперпараметров -> вложенный run MLflow."""
    results = pd.DataFrame(search.cv_results_)
    for i, row in results.iterrows():
        with mlflow.start_run(run_name=f"{model_name}-candidate-{i:02d}", nested=True):
            mlflow.set_tags({"model_type": model_name, "run_kind": "cv_candidate"})
            mlflow.log_params(_strip_prefix(row["params"]))
            mlflow.log_metrics(
                {
                    f"cv_{scoring}_mean": float(row["mean_test_score"]),
                    f"cv_{scoring}_std": float(row["std_test_score"]),
                    "cv_rank": float(row["rank_test_score"]),
                    "mean_fit_time_sec": float(row["mean_fit_time"]),
                }
            )


def log_datasets(data: Dataset) -> None:
    """Связывает run с версиями train/test (MLflow Datasets: путь + хэш данных)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The specified dataset source")
        # Схема датасета — только для информации, пропусков в данных нет.
        warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
        for name, X, y, path, context in (
            ("train", data.X_train, data.y_train, data.train_path, "training"),
            ("test", data.X_test, data.y_test, data.test_path, "evaluation"),
        ):
            dataset = mlflow.data.from_pandas(
                pd.concat([X, y], axis=1), source=str(path), targets=TARGET, name=name
            )
            mlflow.log_input(dataset, context=context)


def _pip_requirements() -> list[str]:
    # Явный список зависимостей: MLflow не тратит время на их автоопределение.
    return [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"mlflow=={mlflow.__version__}",
        f"skops=={skops.__version__}",
    ]


def run_experiment(
    model_name: str,
    model_config: dict[str, Any],
    data: Dataset,
    train_config: dict[str, Any],
    random_state: int,
) -> ExperimentResult:
    """Один эксперимент = один алгоритм с подбором гиперпараметров = один run MLflow."""
    scoring = train_config["scoring"]
    cv_folds = train_config["cv_folds"]
    n_jobs = train_config["n_jobs"]

    search = build_search(
        create_pipeline(model_name, random_state),
        model_config,
        cv_folds=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({"model_type": model_name, "run_kind": "experiment"})
        log_datasets(data)
        mlflow.log_params(
            {
                "model_type": model_name,
                "search_type": model_config.get("search", "grid"),
                "n_iter": model_config.get("n_iter", "all"),
                "cv_folds": cv_folds,
                "scoring": scoring,
                "n_train": len(data.X_train),
                "n_test": len(data.X_test),
                "search_space": json.dumps(model_config.get("params", {})),
            }
        )

        logger.info("[%s] hyperparameter search started", model_name)
        started = time.perf_counter()
        search.fit(data.X_train, data.y_train)
        fit_time = time.perf_counter() - started
        best_pipeline: Pipeline = search.best_estimator_
        best_params = _strip_prefix(search.best_params_)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        logger.info(
            "[%s] best CV %s = %.4f +- %.4f in %.1fs, params=%s",
            model_name,
            scoring,
            search.best_score_,
            cv_std,
            fit_time,
            best_params,
        )

        mlflow.log_params(_prefixed(best_params, "best_"))
        mlflow.log_params(
            _prefixed(best_pipeline.named_steps[CLASSIFIER_STEP].get_params(), "clf_")
        )
        log_search_candidates(search, model_name, scoring)

        # Порог по out-of-fold предсказаниям на train (тест не используется).
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        oof_proba = cross_val_predict(
            best_pipeline, data.X_train, data.y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
        )[:, 1]
        threshold = find_best_threshold(
            data.y_train, oof_proba, train_config.get("threshold_metric", "f1")
        )

        y_proba = best_pipeline.predict_proba(data.X_test)[:, 1]
        metrics_default = compute_metrics(data.y_test, y_proba, threshold=0.5)
        metrics_tuned = compute_metrics(data.y_test, y_proba, threshold=threshold)

        mlflow.log_param("decision_threshold", round(threshold, 4))
        mlflow.log_metrics(
            {
                f"cv_{scoring}_mean": float(search.best_score_),
                f"cv_{scoring}_std": cv_std,
                "oof_roc_auc": float(compute_metrics(data.y_train, oof_proba)["roc_auc"]),
                "fit_time_sec": fit_time,
                **_prefixed(metrics_default, "test_"),
                **_prefixed(
                    {k: metrics_tuned[k] for k in ("precision", "recall", "f1", "accuracy")},
                    "test_tuned_",
                ),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            save_evaluation_plots(
                best_pipeline, data.X_test, data.y_test, y_proba, threshold, tmp_dir, model_name
            )
            pd.DataFrame(search.cv_results_).drop(columns=["params"]).to_csv(
                tmp_dir / "cv_results.csv", index=False
            )
            report = classification_report(
                data.y_test,
                (y_proba >= threshold).astype(int),
                target_names=["no_default", "default"],
                digits=4,
            )
            (tmp_dir / "classification_report.txt").write_text(
                f"Threshold: {threshold:.4f}\n\n{report}", encoding="utf-8"
            )
            mlflow.log_artifacts(str(tmp_dir), artifact_path="evaluation")

        # float64: в схеме MLflow целые колонки не допускают пропусков.
        input_example = data.X_train.head(5).astype("float64")
        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            signature=infer_signature(input_example, best_pipeline.predict(input_example)),
            input_example=input_example,
            pip_requirements=_pip_requirements(),
            code_paths=[str(PROJECT_ROOT / "src")],
            # MLflow сохраняет sklearn-модели в безопасном формате skops и требует
            # явно перечислить "нестандартные" типы (наш FeatureEngineer и т.п.).
            # Модель создана нами же, поэтому доверяем всем её типам.
            skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(best_pipeline)),
        )

    return ExperimentResult(
        model_name=model_name,
        run_id=run.info.run_id,
        model_uri=model_info.model_uri,
        pipeline=best_pipeline,
        cv_score=float(search.best_score_),
        cv_std=cv_std,
        threshold=threshold,
        best_params=best_params,
        test_metrics=metrics_tuned,
        test_metrics_default=metrics_default,
        fit_time_sec=fit_time,
    )


def summarize(results: list[ExperimentResult], scoring: str) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model_name,
                f"cv_{scoring}_mean": round(r.cv_score, 4),
                f"cv_{scoring}_std": round(r.cv_std, 4),
                "test_roc_auc": round(r.test_metrics["roc_auc"], 4),
                "test_gini": round(r.test_metrics["gini"], 4),
                "test_ks": round(r.test_metrics["ks"], 4),
                "test_pr_auc": round(r.test_metrics["pr_auc"], 4),
                "threshold": round(r.threshold, 4),
                "test_precision": round(r.test_metrics["precision"], 4),
                "test_recall": round(r.test_metrics["recall"], 4),
                "test_f1": round(r.test_metrics["f1"], 4),
                "test_brier": round(r.test_metrics["brier"], 4),
                "fit_time_sec": round(r.fit_time_sec, 1),
                "mlflow_run_id": r.run_id,
            }
        )
    return pd.DataFrame(rows).sort_values(f"cv_{scoring}_mean", ascending=False)


def plot_model_comparison(summary: pd.DataFrame, scoring: str, path: Path) -> Path:
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    x = np.arange(len(summary))
    cv_mean, cv_std = summary[f"cv_{scoring}_mean"], summary[f"cv_{scoring}_std"]
    ax.bar(x - 0.2, cv_mean, 0.4, yerr=cv_std, capsize=4, label=f"CV {scoring} (mean ± std)")
    ax.bar(x + 0.2, summary["test_roc_auc"], 0.4, label="Test ROC-AUC")
    for i, (cv, std, test) in enumerate(zip(cv_mean, cv_std, summary["test_roc_auc"])):
        ax.text(i - 0.2, cv + std + 0.002, f"{cv:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, test + 0.003, f"{test:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, summary["model"], rotation=20, ha="right")
    low = min(cv_mean.min(), summary["test_roc_auc"].min())
    high = max((cv_mean + cv_std).max(), summary["test_roc_auc"].max())
    ax.set_ylim(low - 0.03, high + 0.03)
    ax.set(title="Model comparison (best model = highest CV score)", ylabel="ROC-AUC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    return path


def register_champion(result: ExperimentResult, registered_name: str) -> str | None:
    """Регистрирует лучшую модель в MLflow Model Registry с алиасом ``champion``."""
    try:
        version = mlflow.register_model(result.model_uri, registered_name)
        client = MlflowClient()
        client.set_registered_model_alias(registered_name, "champion", version.version)
        client.set_tag(result.run_id, "champion", "true")
        logger.info("Registered %s version %s (alias 'champion')", registered_name, version.version)
        return str(version.version)
    except Exception as exc:  # реестр недоступен (например, файловый бэкенд без БД)
        logger.warning("Model registration skipped: %s", exc)
        return None


def save_outputs(
    best: ExperimentResult,
    summary: pd.DataFrame,
    data: Dataset,
    train_config: dict[str, Any],
    registered_version: str | None,
) -> None:
    """Сохраняет модель и отчёты, которые версионирует DVC."""
    model_path = resolve_path(train_config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipeline, model_path)

    scoring = train_config["scoring"]
    metadata = {
        "model_name": best.model_name,
        "threshold": round(best.threshold, 6),
        "threshold_metric": train_config.get("threshold_metric", "f1"),
        f"cv_{scoring}_mean": round(best.cv_score, 6),
        "best_params": best.best_params,
        "test_metrics": {k: round(v, 6) for k, v in best.test_metrics.items()},
        "features": RAW_FEATURES,
        "train_rows": len(data.X_train),
        "sklearn_version": sklearn.__version__,
        "mlflow_run_id": best.run_id,
        "mlflow_model_uri": best.model_uri,
        "registered_model_version": registered_version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = resolve_path(train_config["meta_path"])
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # Распределение скоров на train — эталон для мониторинга дрифта (PSI).
    reference_scores = best.pipeline.predict_proba(data.X_train)[:, 1]
    pd.DataFrame({"score": np.round(reference_scores, 6)}).to_csv(
        resolve_path(train_config["reference_scores_path"]), index=False
    )

    metrics_path = resolve_path(train_config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_model": best.model_name,
        f"cv_{scoring}": round(best.cv_score, 4),
        "threshold": round(best.threshold, 4),
        "test": {k: round(v, 4) for k, v in best.test_metrics.items()},
        "test_threshold_0_5": {
            k: round(best.test_metrics_default[k], 4)
            for k in ("precision", "recall", "f1", "accuracy")
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_path = resolve_path(train_config["summary_path"])
    summary.to_csv(summary_path, index=False)

    y_proba = best.pipeline.predict_proba(data.X_test)[:, 1]
    roc_points(data.y_test, y_proba).to_csv(
        resolve_path(train_config["roc_points_path"]), index=False
    )
    figures_dir = resolve_path(train_config["figures_dir"])
    save_evaluation_plots(
        best.pipeline,
        data.X_test,
        data.y_test,
        y_proba,
        best.threshold,
        figures_dir,
        best.model_name,
    )
    plot_model_comparison(summary, scoring, figures_dir / "model_comparison.png")
    logger.info("Saved model to %s, metrics to %s", model_path, metrics_path)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    parser = argparse.ArgumentParser(description="Train PD models with MLflow tracking")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated subset of models to train")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow Model Registry")
    args = parser.parse_args(argv)

    params = load_params(args.params)
    train_config = params["train"]
    if args.cv_folds:
        train_config["cv_folds"] = args.cv_folds
    random_state = params["base"]["random_state"]

    model_names = args.models.split(",") if args.models else list(params["models"])
    data = load_dataset(args.processed_dir or params["data"]["processed_dir"])
    logger.info("Train: %d rows, test: %d rows", len(data.X_train), len(data.X_test))

    setup_mlflow(params["mlflow"])

    results = [
        run_experiment(name, params["models"][name], data, train_config, random_state)
        for name in model_names
    ]
    summary = summarize(results, train_config["scoring"])
    logger.info("Experiments summary:\n%s", summary.drop(columns=["mlflow_run_id"]).to_string())

    best = max(results, key=lambda r: r.cv_score)
    logger.info("Best model by CV: %s (CV ROC-AUC %.4f)", best.model_name, best.cv_score)
    registered_version = (
        None
        if args.no_register
        else register_champion(best, params["mlflow"]["registered_model_name"])
    )
    save_outputs(best, summary, data, train_config, registered_version)


if __name__ == "__main__":
    main()
