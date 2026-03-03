from datetime import datetime, timezone

from fastapi import APIRouter

from app.observability.metrics import record_prediction
from app.schemas.risk import RiskRequest, RiskResponse
from app.services.ml_inference import ChurnRiskModelService

router = APIRouter()
model_service = ChurnRiskModelService()


@router.get("/healthz", tags=["System"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict-risk", response_model=RiskResponse, tags=["Risk"])
def predict_risk(payload: RiskRequest) -> RiskResponse:
    result = model_service.predict(payload)
    record_prediction(result["risk_category"])
    return RiskResponse(
        risk_category=result["risk_category"],
        churn_probability=result["churn_probability"],
        churn_prediction=result["churn_prediction"],
        reasons=result["reasons"],
        ticket_count_last_30_days=result["ticket_count_last_30_days"],
        model_version=result["model_version"],
        evaluated_at=datetime.now(timezone.utc),
    )
