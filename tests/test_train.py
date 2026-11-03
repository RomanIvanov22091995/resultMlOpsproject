"""Интеграционный smoke-тест обучения: данные -> MLflow -> артефакты модели."""

import json

import mlflow
import pytest
import yaml

from src.config import PARAMS_PATH
from src.data.make_dataset import main as make_dataset
from src.models.predict import load_model
from src.models.train import main as train


@pytest.fixture
def quick_params(tmp_path):
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    params["models"] = {
        "logistic_regression": {"search": "grid", "params": {"C": [0.1, 1.0]}},
        "hist_gradient_boosting": {"search": "random", "n_iter": 2, "params": {"max_iter": [50]}},
    }
    params["train"].update(
        {
            "cv_folds": 2,
            "n_jobs": 1,
            "model_path": str(tmp_path / "model.pkl"),
            "meta_path": str(tmp_path / "model_meta.json"),
            "reference_scores_path": str(tmp_path / "reference_scores.csv"),
            "metrics_path": str(tmp_path / "metrics.json"),
            "figures_dir": str(tmp_path / "figures"),
            "roc_points_path": str(tmp_path / "roc.csv"),
            "summary_path": str(tmp_path / "summary.csv"),
        }
    )
    params["mlflow"].update(
        {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "artifact_location": str(tmp_path / "mlruns"),
            "experiment_name": "test-experiment",
        }
    )
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path, params


def test_training_end_to_end(tmp_path, quick_params, monkeypatch):
    params_path, params = quick_params
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    processed = tmp_path / "processed"
    make_dataset(["--input", "data/sample/credit_sample.csv", "--output-dir", str(processed)])

    train(["--params", str(params_path), "--processed-dir", str(processed)])

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["best_model"] in params["models"]
    for name in ("roc_auc", "gini", "ks", "precision", "recall", "f1"):
        assert 0.0 <= metrics["test"][name] <= 1.0
    assert metrics["test"]["roc_auc"] > 0.6

    for figure in ("roc_curve", "pr_curve", "confusion_matrix", "model_comparison"):
        assert (tmp_path / "figures" / f"{figure}.png").exists()

    model = load_model(tmp_path / "model.pkl", tmp_path / "model_meta.json")
    assert model.name == metrics["best_model"]

    runs = mlflow.search_runs(experiment_names=["test-experiment"])
    parents = runs[runs["tags.run_kind"] == "experiment"]
    assert len(parents) == 2
    assert "metrics.test_roc_auc" in runs.columns
    assert (runs["tags.run_kind"] == "cv_candidate").sum() == 3
    champion = mlflow.MlflowClient().get_model_version_by_alias("CreditDefaultModel", "champion")
    assert str(champion.version) == "1"
