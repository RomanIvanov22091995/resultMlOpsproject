"""Метрики качества бинарной классификации и графики для отчётов.

Помимо стандартных ROC-AUC / Precision / Recall / F1 считаются метрики,
принятые в кредитном скоринге:

* Gini = 2 * ROC-AUC - 1;
* KS (статистика Колмогорова–Смирнова) — максимальное расстояние между
  распределениями скоров "хороших" и "плохих" клиентов;
* Brier score — качество калибровки вероятностей (PD должна быть похожа на
  реальную частоту дефолтов).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def ks_statistic(y_true, y_proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Все метрики на одном наборе предсказанных вероятностей."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_proba)
    return {
        "roc_auc": float(roc_auc),
        "gini": float(2 * roc_auc - 1),
        "ks": ks_statistic(y_true, y_proba),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
    }


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Порог отсечения, максимизирующий F1 (``f1``) или J-статистику Юдена (``youden``)."""
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        denominator = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        return float(thresholds[int(np.argmax(f1))])
    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return float(min(thresholds[int(np.argmax(tpr - fpr))], 1.0))
    raise ValueError(f"Unknown threshold metric '{metric}', expected 'f1' or 'youden'")


def roc_points(y_true, y_proba) -> pd.DataFrame:
    """Точки ROC-кривой (для ``dvc plots``)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return pd.DataFrame({"fpr": np.round(fpr, 5), "tpr": np.round(tpr, 5)})


def _new_axes(figsize: tuple[float, float]):
    # Объектный API matplotlib (без pyplot) не требует GUI-бэкенда:
    # работает одинаково локально, в Docker и в CI.
    fig = Figure(figsize=figsize)
    return fig, fig.subplots()


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_roc_curve(y_true, y_proba, path: Path, title: str = "ROC curve") -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = _new_axes((6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {auc:.4f} (Gini = {2 * auc - 1:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random model")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_precision_recall_curve(y_true, y_proba, path: Path, title: str = "PR curve") -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = _new_axes((6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC (AP) = {ap:.4f}")
    ax.axhline(np.mean(y_true), ls="--", color="grey", label="Default rate (baseline)")
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "Confusion matrix") -> Path:
    fig, ax = _new_axes((5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No default", "Default"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    return _save(fig, path)


def plot_calibration_curve(y_true, y_proba, path: Path, title: str = "Calibration") -> Path:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = _new_axes((6, 5))
    ax.plot(prob_pred, prob_true, "o-", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Predicted PD", ylabel="Observed default rate", title=title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_score_distribution(
    y_true, y_proba, path: Path, threshold: float | None = None, title: str = "Score distribution"
) -> Path:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fig, ax = _new_axes((7, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.6, density=True, label="No default (0)")
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.6, density=True, label="Default (1)")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", label=f"Threshold = {threshold:.3f}")
    ax.set(xlabel="Predicted probability of default", ylabel="Density", title=title)
    ax.legend()
    return _save(fig, path)


def feature_importance(pipeline: Pipeline, X: pd.DataFrame, y, random_state: int = 42) -> pd.Series:
    """Важность признаков: встроенная у деревьев, |coef| у логрегрессии,
    permutation importance (по сырым признакам) для остальных моделей."""
    classifier = pipeline.named_steps["classifier"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        sample = X.sample(n=min(len(X), 3000), random_state=random_state)
        y_sample = pd.Series(np.asarray(y), index=X.index).loc[sample.index]
        result = permutation_importance(
            pipeline,
            sample,
            y_sample,
            scoring="roc_auc",
            n_repeats=5,
            random_state=random_state,
            n_jobs=1,
        )
        names, values = sample.columns, result.importances_mean
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_feature_importance(importances: pd.Series, path: Path, top_n: int = 20) -> Path:
    top = importances.head(top_n)[::-1]
    fig, ax = _new_axes((7, 0.32 * len(top) + 1.2))
    ax.barh(top.index, top.values)
    ax.set(title=f"Top-{len(top)} feature importances", xlabel="Importance")
    return _save(fig, path)


def save_evaluation_plots(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test,
    y_proba,
    threshold: float,
    out_dir: Path,
    title: str = "",
) -> list[Path]:
    """Все графики оценки модели в одну папку."""
    out_dir = Path(out_dir)
    prefix = f"{title}: " if title else ""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    importances = feature_importance(pipeline, X_test, y_test)
    return [
        plot_roc_curve(y_test, y_proba, out_dir / "roc_curve.png", f"{prefix}ROC curve (test)"),
        plot_precision_recall_curve(
            y_test, y_proba, out_dir / "pr_curve.png", f"{prefix}Precision-Recall (test)"
        ),
        plot_confusion_matrix(
            y_test,
            y_pred,
            out_dir / "confusion_matrix.png",
            f"{prefix}Confusion matrix (thr={threshold:.3f})",
        ),
        plot_calibration_curve(
            y_test, y_proba, out_dir / "calibration_curve.png", f"{prefix}Calibration (test)"
        ),
        plot_score_distribution(
            y_test, y_proba, out_dir / "score_distribution.png", threshold, f"{prefix}Scores (test)"
        ),
        plot_feature_importance(importances, out_dir / "feature_importance.png"),
    ]
