"""Имитация потока новых заявок и мониторинг дрифта.

Скрипт берёт часть тестовой выборки ("новые клиенты"), отправляет каждую
запись в REST API (``POST /predict``), собирает ответы и сравнивает
распределения с обучающими данными через PSI:

* PSI скора модели (вероятности дефолта) — главный индикатор стабильности;
* PSI каждого входного признака — помогает понять, *что именно* сдвинулось.

Опция ``--drift`` искусственно портит данные (снижение лимитов, ухудшение
платёжной дисциплины и т.д.), чтобы показать, как мониторинг ловит дрифт.

Примеры::

    python -m src.monitoring.simulate_traffic --api-url http://localhost:8000
    python -m src.monitoring.simulate_traffic --api-url http://localhost:8000 --drift
    python -m src.monitoring.simulate_traffic --mode local --drift   # без API
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

from src.config import RAW_FEATURES, TARGET, load_params, resolve_path, setup_logging
from src.monitoring.drift import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    calculate_psi,
    feature_drift_report,
    psi_status,
)

logger = logging.getLogger(__name__)

KEY_FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"]


def apply_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Сценарий "кризис": лимиты урезаны, клиенты старше, просрочки растут."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["LIMIT_BAL"] = (df["LIMIT_BAL"] * 0.6).clip(lower=10_000).round(-3)
    df["AGE"] = (df["AGE"] + 7).clip(upper=100)
    worsened = rng.random(len(df)) < 0.35
    df.loc[worsened, "PAY_0"] = (df.loc[worsened, "PAY_0"].clip(lower=0) + 2).clip(upper=9)
    df["BILL_AMT1"] = (df["BILL_AMT1"] * 1.3).round()
    return df


def score_via_api(df: pd.DataFrame, api_url: str, timeout: float = 10.0) -> tuple[np.ndarray, list]:
    """Отправляет клиентов по одному в ``/predict`` (как реальный поток заявок)."""
    records = json.loads(df[RAW_FEATURES].to_json(orient="records"))
    probabilities, latencies = [], []
    with requests.Session() as session:
        health = session.get(f"{api_url}/health", timeout=timeout)
        health.raise_for_status()
        for i, record in enumerate(records, start=1):
            started = time.perf_counter()
            response = session.post(f"{api_url}/predict", json=record, timeout=timeout)
            latencies.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()
            probabilities.append(response.json()["default_probability"])
            if i % 200 == 0:
                logger.info("Sent %d/%d requests", i, len(records))
    return np.asarray(probabilities), latencies


def score_locally(df: pd.DataFrame) -> np.ndarray:
    from src.models.predict import load_model

    return load_model().predict_proba(df)


def plot_score_distributions(reference: np.ndarray, current: np.ndarray, psi: float, path: Path):
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    bins = np.linspace(0, 1, 41)
    ax.hist(reference, bins=bins, density=True, alpha=0.55, label="Reference (train)")
    ax.hist(current, bins=bins, density=True, alpha=0.55, label="Current (new data)")
    ax.set(
        title=f"Score distribution, PSI = {psi:.3f} ({psi_status(psi)})",
        xlabel="Predicted probability of default",
        ylabel="Density",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def plot_feature_psi(report: pd.DataFrame, path: Path):
    top = report.head(15)[::-1]
    colors = [
        "#d62728" if p >= PSI_SIGNIFICANT else "#ff7f0e" if p >= PSI_MODERATE else "#2ca02c"
        for p in top["psi"]
    ]
    fig = Figure(figsize=(7, 5.5))
    ax = fig.subplots()
    ax.barh(top["feature"], top["psi"], color=colors)
    ax.axvline(PSI_MODERATE, ls="--", color="#ff7f0e", label=f"moderate ({PSI_MODERATE})")
    ax.axvline(PSI_SIGNIFICANT, ls="--", color="#d62728", label=f"significant ({PSI_SIGNIFICANT})")
    ax.set(title="PSI by feature (top-15)", xlabel="PSI")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def write_markdown(summary: dict, report: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Data drift report",
        "",
        f"- Scenario: **{summary['scenario']}**, samples: {summary['n_samples']}, "
        f"mode: {summary['mode']}",
        f"- Score PSI: **{summary['score_psi']:.4f} ({summary['score_status']})**",
        f"- Features with significant drift: {summary['significant_features'] or 'none'}",
        f"- Features with moderate drift: {summary['moderate_features'] or 'none'}",
        f"- ROC-AUC on new data: {summary['current_roc_auc']:.4f} "
        f"(reference test ROC-AUC: {summary['reference_test_roc_auc']})",
        f"- Mean PD: reference {summary['reference_mean_pd']:.4f} -> "
        f"current {summary['current_mean_pd']:.4f}",
    ]
    if summary.get("latency_ms_p50") is not None:
        lines.append(
            f"- API latency: p50 {summary['latency_ms_p50']:.1f} ms, "
            f"p95 {summary['latency_ms_p95']:.1f} ms"
        )
    lines += [
        "",
        "PSI thresholds: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant.",
        "",
        "| feature | type | PSI | status | reference mean | current mean |",
        "|---|---|---|---|---|---|",
    ]
    for row in report.itertuples():
        lines.append(
            f"| {row.feature} | {row.type} | {row.psi:.4f} | {row.status} | "
            f"{row.reference_mean} | {row.current_mean} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    params = load_params()
    parser = argparse.ArgumentParser(description="Simulate new traffic and monitor data drift")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["api", "local"], default="api")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--drift", action="store_true", help="Inject synthetic drift")
    parser.add_argument("--seed", type=int, default=params["base"]["random_state"])
    parser.add_argument("--output-dir", default="reports/monitoring")
    parser.add_argument(
        "--fail-on-drift", action="store_true", help="Exit with code 1 on significant drift"
    )
    args = parser.parse_args(argv)

    processed_dir = resolve_path(params["data"]["processed_dir"])
    reference = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    reference_scores = pd.read_csv(resolve_path(params["train"]["reference_scores_path"]))["score"]
    meta_path = resolve_path(params["train"]["meta_path"])
    reference_auc = json.loads(meta_path.read_text())["test_metrics"]["roc_auc"]

    current = test.sample(n=min(args.n_samples, len(test)), random_state=args.seed)
    current = current.reset_index(drop=True)
    if args.drift:
        current = apply_drift(current, args.seed)
    scenario = "synthetic drift" if args.drift else "no drift (holdout sample)"
    logger.info("Scenario: %s, %d new clients, mode=%s", scenario, len(current), args.mode)

    latencies = None
    if args.mode == "api":
        scores, latencies = score_via_api(current, args.api_url)
    else:
        scores = score_locally(current)

    score_psi = calculate_psi(reference_scores, scores)
    report = feature_drift_report(reference, current, RAW_FEATURES)
    summary = {
        "scenario": scenario,
        "mode": args.mode,
        "n_samples": len(current),
        "score_psi": round(score_psi, 4),
        "score_status": psi_status(score_psi),
        "key_features_psi": {
            row.feature: row.psi for row in report.itertuples() if row.feature in KEY_FEATURES
        },
        "significant_features": report.loc[report["status"] == "significant", "feature"].tolist(),
        "moderate_features": report.loc[report["status"] == "moderate", "feature"].tolist(),
        "reference_mean_pd": round(float(reference_scores.mean()), 4),
        "current_mean_pd": round(float(scores.mean()), 4),
        "current_roc_auc": round(float(roc_auc_score(current[TARGET], scores)), 4),
        "reference_test_roc_auc": round(float(reference_auc), 4),
        "latency_ms_p50": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
    }

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "drift_report.json").write_text(
        json.dumps({"summary": summary, "features": report.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    report.to_csv(output_dir / "feature_psi.csv", index=False)
    write_markdown(summary, report, output_dir / "drift_report.md")
    plot_score_distributions(
        reference_scores.to_numpy(), scores, score_psi, output_dir / "score_psi.png"
    )
    plot_feature_psi(report, output_dir / "feature_psi.png")

    logger.info("Score PSI = %.4f (%s)", score_psi, summary["score_status"])
    logger.info("Top drifted features:\n%s", report.head(8).to_string(index=False))
    logger.info("Reports saved to %s", output_dir)

    drift_detected = summary["score_status"] == "significant" or summary["significant_features"]
    if drift_detected:
        logger.warning("SIGNIFICANT DRIFT DETECTED: consider retraining the model")
        if args.fail_on_drift:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
