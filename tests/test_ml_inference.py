from app.schemas.risk import CustomerProfile, RiskRequest, TicketRecord
from app.services.ml_inference import ChurnRiskModelService


def test_ml_inference_fallback_predicts_valid_output():
    service = ChurnRiskModelService(model_path="artifacts/does-not-exist.joblib")

    payload = RiskRequest(
        customer=CustomerProfile(
            customer_id="C-500",
            contract_type="Month-to-Month",
            monthly_charges_current=95.0,
            monthly_charges_previous=80.0,
            tenure_months=6,
            internet_service="Fiber optic",
            payment_method="Electronic check",
        ),
        tickets=[
            TicketRecord(
                opened_at="2026-02-28T10:00:00Z",
                category="complaint",
                is_complaint=True,
                sentiment_score=-0.8,
            ),
            TicketRecord(
                opened_at="2026-02-27T10:00:00Z",
                category="billing",
                is_complaint=False,
                sentiment_score=-0.2,
            ),
        ],
    )

    result = service.predict(payload)
    assert result["risk_category"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] in {"Yes", "No"}
    assert isinstance(result["ticket_count_last_30_days"], int)
