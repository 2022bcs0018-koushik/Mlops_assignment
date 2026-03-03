from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app
from app.observability.metrics import reset_metrics

client = TestClient(app)


def setup_function():
    reset_metrics()


def test_predict_risk_endpoint_returns_ml_response(monkeypatch):
    def fake_predict(_payload):
        return {
            "risk_category": "High",
            "churn_probability": 0.87,
            "churn_prediction": "Yes",
            "reasons": ["High support ticket frequency in the last 30 days."],
            "ticket_count_last_30_days": 6,
            "model_version": "test-model-v1",
        }

    monkeypatch.setattr(routes.model_service, "predict", fake_predict)

    payload = {
        "customer": {
            "customer_id": "C-111",
            "contract_type": "Month-to-Month",
            "monthly_charges_current": 110.0,
            "monthly_charges_previous": 99.0,
            "tenure_months": 24,
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        },
        "tickets": [
            {
                "opened_at": "2026-02-20T10:00:00Z",
                "category": "complaint",
                "is_complaint": True,
                "sentiment_score": -0.7,
            }
        ],
    }
    response = client.post("/predict-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_category"] == "High"
    assert body["churn_prediction"] == "Yes"
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == "test-model-v1"
    assert "reasons" in body


def test_health_and_metrics_endpoints():
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "churn_requests_total" in metrics.text
