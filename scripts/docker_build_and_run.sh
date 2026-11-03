#!/usr/bin/env bash
# Сборка Docker-образа PD-модели, запуск контейнера и проверка API.
# Использование: ./scripts/docker_build_and_run.sh [порт]
set -euo pipefail

IMAGE="credit-scoring-api:latest"
CONTAINER="credit-scoring-api"
PORT="${1:-8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT"

if [ ! -f models/credit_default_model.pkl ]; then
  echo "Model not found. Run 'dvc repro' (or 'dvc pull') first." >&2
  exit 1
fi

echo ">>> Building image $IMAGE"
docker build -t "$IMAGE" .

echo ">>> (Re)starting container $CONTAINER on port $PORT"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p "$PORT:8000" "$IMAGE"

echo ">>> Smoke test"
"$PYTHON" scripts/smoke_test_api.py --url "http://localhost:$PORT" --wait 90

echo ">>> API is running: http://localhost:$PORT/docs"
echo ">>> Stop it with: docker rm -f $CONTAINER"
