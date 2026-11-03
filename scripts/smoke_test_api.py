"""Smoke-тест запущенного API: /health, /predict, /predict_batch и валидация входа.

    python scripts/smoke_test_api.py --url http://localhost:8000

Код возврата 0 — всё работает, 1 — есть проблемы.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

CLIENT = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 46990,
    "BILL_AMT2": 48233,
    "BILL_AMT3": 49291,
    "BILL_AMT4": 28314,
    "BILL_AMT5": 28959,
    "BILL_AMT6": 29547,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 2019,
    "PAY_AMT3": 1200,
    "PAY_AMT4": 1100,
    "PAY_AMT5": 1069,
    "PAY_AMT6": 1000,
}
GOOD_CLIENT = {**CLIENT, "LIMIT_BAL": 500000, "PAY_0": -1, "PAY_2": -1, "PAY_AMT1": 40000}


def wait_for_api(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"API at {url} did not become healthy in {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=60, help="Seconds to wait for /health")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    wait_for_api(url, args.wait)
    print("GET /health ->", requests.get(f"{url}/health", timeout=5).json())
    print("GET /model-info ->", requests.get(f"{url}/model-info", timeout=5).json()["model_name"])

    risky = requests.post(f"{url}/predict", json=CLIENT, timeout=5)
    risky.raise_for_status()
    print("POST /predict (risky client) ->", json.dumps(risky.json()))

    good = requests.post(f"{url}/predict", json=GOOD_CLIENT, timeout=5)
    good.raise_for_status()
    print("POST /predict (reliable client) ->", json.dumps(good.json()))

    batch = requests.post(
        f"{url}/predict_batch", json={"clients": [CLIENT, GOOD_CLIENT]}, timeout=5
    )
    batch.raise_for_status()
    print("POST /predict_batch ->", len(batch.json()["predictions"]), "predictions")

    invalid = requests.post(f"{url}/predict", json={**CLIENT, "AGE": 150}, timeout=5)
    print("POST /predict (AGE=150) ->", invalid.status_code)

    checks = {
        "risky client has higher PD": risky.json()["default_probability"]
        > good.json()["default_probability"],
        "invalid input rejected with 422": invalid.status_code == 422,
    }
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
