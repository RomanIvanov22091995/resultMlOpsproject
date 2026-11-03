import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from tests.conftest import TEST_THRESHOLD


@pytest.fixture
def client(trained_model_files, monkeypatch):
    model_path, meta_path = trained_model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_META_PATH", str(meta_path))
    with TestClient(app) as test_client:  # with -> выполняется lifespan (загрузка модели)
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setenv("MODEL_META_PATH", str(tmp_path / "missing.json"))
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "alive" in response.json()["message"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model_name"] == "logistic_regression"
    assert info["threshold"] == TEST_THRESHOLD


def test_predict_returns_class_and_probability(client, client_payload):
    response = client.post("/predict", json=client_payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"default_prediction", "default_probability", "threshold", "model_name"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["default_prediction"] == int(body["default_probability"] >= TEST_THRESHOLD)


def test_predict_risky_client_has_higher_pd(client, client_payload):
    reliable = {**client_payload, "PAY_0": -1, "PAY_2": -1, "LIMIT_BAL": 500_000}
    risky = {**client_payload, "PAY_0": 3, "PAY_2": 3, "PAY_3": 2}
    p_reliable = client.post("/predict", json=reliable).json()["default_probability"]
    p_risky = client.post("/predict", json=risky).json()["default_probability"]
    assert p_risky > p_reliable


@pytest.mark.parametrize(
    "field, value",
    [("AGE", 150), ("AGE", 10), ("SEX", 3), ("EDUCATION", 7), ("PAY_0", 12), ("LIMIT_BAL", -5)],
)
def test_predict_rejects_invalid_values(client, client_payload, field, value):
    response = client.post("/predict", json={**client_payload, field: value})
    assert response.status_code == 422


def test_predict_rejects_missing_and_extra_fields(client, client_payload):
    missing = {k: v for k, v in client_payload.items() if k != "PAY_AMT6"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**client_payload, "ID": 1}).status_code == 422


def test_predict_batch(client, client_payload):
    response = client.post("/predict_batch", json={"clients": [client_payload] * 3})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert len({p["default_probability"] for p in predictions}) == 1


def test_predict_batch_rejects_empty_list(client):
    assert client.post("/predict_batch", json={"clients": []}).status_code == 422


def test_service_without_model_returns_503(client_without_model, client_payload):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=client_payload).status_code == 503
