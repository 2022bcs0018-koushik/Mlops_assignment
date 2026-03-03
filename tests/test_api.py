from fastapi.testclient import TestClient

from app.main import app
from app.observability.metrics import reset_metrics

client = TestClient(app)


def setup_function():
    reset_metrics()


def test_predict_risk_endpoint_returns_high():
    payload = {
        "customer": {
            "customer_id": "C-111",
            "contract_type": "Month-to-Month",
            "monthly_charges_current": 110.0,
            "monthly_charges_previous": 99.0,
        },
        "tickets": [
            {
                "opened_at": "2026-02-20T10:00:00Z",
                "category": "complaint",
                "is_complaint": True,
            }
        ],
    }
    response = client.post("/predict-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_category"] == "High"
    assert "reasons" in body


def test_health_and_metrics_endpoints():
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "churn_requests_total" in metrics.text
