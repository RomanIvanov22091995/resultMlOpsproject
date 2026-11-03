"""Создаёт небольшую стратифицированную выборку сырых данных для CI и тестов.

Полный датасет версионируется DVC и в git не попадает, а маленький sample
(2000 строк) лежит в git — на нём GitHub Actions запускает валидацию
Great Expectations, тесты и пробное обучение.

    python scripts/create_sample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "default.payment.next.month"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=PROJECT_ROOT / "data/raw/UCI_Credit_Card.csv")
    parser.add_argument("--output", default=PROJECT_ROOT / "data/sample/credit_sample.csv")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sample, _ = train_test_split(
        df, train_size=args.size, stratify=df[TARGET], random_state=args.random_state
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.sort_values("ID").to_csv(output, index=False)
    print(f"Saved {len(sample)} rows (default rate {sample[TARGET].mean():.3f}) to {output}")


if __name__ == "__main__":
    main()
