# Образ PD-модели: содержит все зависимости для инференса (FastAPI)
# и для обучения/валидации (sklearn, MLflow, Great Expectations).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/credit_default_model.pkl \
    MODEL_META_PATH=/app/models/model_meta.json

WORKDIR /app

# Сначала только зависимости: этот слой кэшируется и не пересобирается
# при изменении кода.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml .
COPY src/ src/
COPY models/ models/

# Не запускаем сервис от root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# В продакшне без --reload (он нужен только при разработке).
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
