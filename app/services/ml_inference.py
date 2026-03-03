from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.ml.config import DEFAULT_RISK_THRESHOLDS, MODEL_FEATURE_COLUMNS
from app.ml.feature_engineering import (
    CustomerObservation,
    TicketObservation,
    build_feature_vector,
)
from app.schemas.risk import RiskRequest

logger = logging.getLogger(__name__)


class ChurnRiskModelService:
    """Loads model artifact and provides churn probability + risk inference."""

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/churn_model.joblib"))
        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] = {
            "model_version": "fallback-v1",
            "risk_thresholds": DEFAULT_RISK_THRESHOLDS,
            "feature_columns": MODEL_FEATURE_COLUMNS,
        }
        self._load_artifact_if_present()

    def _load_artifact_if_present(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                "Model artifact not found at '%s'. Falling back to heuristic scorer.",
                self.model_path,
            )
            return

        artifact = joblib.load(self.model_path)
        self.pipeline = artifact.get("pipeline")
        self.metadata.update(artifact.get("metadata", {}))
        logger.info(
            "Loaded model artifact from '%s' (version=%s).",
            self.model_path,
            self.metadata.get("model_version", "unknown"),
        )

    def _heuristic_probability(self, features: dict[str, float | int | str]) -> float:
        z = -1.2
        z += 0.22 * float(features["ticket_count_30d"])
        z += 0.08 * max(0.0, float(features["monthly_charge_change"]))
        z += 0.45 * float(features["complaint_ticket_count"])
        z += 0.60 * max(0.0, -float(features["avg_ticket_sentiment"]))
        probability = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, probability))

    def _predict_probability(self, features: dict[str, float | int | str]) -> float:
        if self.pipeline is None:
            return self._heuristic_probability(features)

        frame = pd.DataFrame([features], columns=MODEL_FEATURE_COLUMNS)
        probability = float(self.pipeline.predict_proba(frame)[0][1])
        return max(0.0, min(1.0, probability))

    def _risk_from_probability(self, probability: float) -> str:
        thresholds = self.metadata.get("risk_thresholds", DEFAULT_RISK_THRESHOLDS)
        high_threshold = float(thresholds.get("high", DEFAULT_RISK_THRESHOLDS["high"]))
        medium_threshold = float(
            thresholds.get("medium", DEFAULT_RISK_THRESHOLDS["medium"])
        )

        if probability >= high_threshold:
            return "High"
        if probability >= medium_threshold:
            return "Medium"
        return "Low"

    def _reasons(
        self,
        features: dict[str, float | int | str],
        probability: float,
        risk_category: str,
    ) -> list[str]:
        reasons: list[str] = []

        if float(features["ticket_count_30d"]) >= 5:
            reasons.append("High support ticket frequency in the last 30 days.")
        if float(features["complaint_ticket_count"]) >= 1:
            reasons.append("Complaint ticket pattern detected.")
        if float(features["monthly_charge_change"]) > 0:
            reasons.append("Monthly charges increased compared to the previous cycle.")
        if float(features["avg_ticket_sentiment"]) < -0.2:
            reasons.append("Negative ticket sentiment observed.")

        if not reasons:
            reasons.append(
                f"Model probability {probability:.2f} mapped to {risk_category} risk."
            )
        return reasons

    def _to_features(self, payload: RiskRequest) -> dict[str, float | int | str]:
        customer = payload.customer
        customer_obs = CustomerObservation(
            customer_id=customer.customer_id,
            contract_type=customer.contract_type,
            monthly_charges_current=customer.monthly_charges_current,
            monthly_charges_previous=customer.monthly_charges_previous,
            tenure_months=customer.tenure_months,
            total_charges=customer.total_charges,
            internet_service=customer.internet_service,
            payment_method=customer.payment_method,
        )

        tickets = [
            TicketObservation(
                opened_at=ticket.opened_at,
                category=ticket.category,
                is_complaint=ticket.is_complaint,
                sentiment_score=ticket.sentiment_score,
            )
            for ticket in payload.tickets
        ]
        return build_feature_vector(customer_obs, tickets)

    def predict(self, payload: RiskRequest) -> dict[str, Any]:
        features = self._to_features(payload)
        probability = self._predict_probability(features)
        risk_category = self._risk_from_probability(probability)
        reasons = self._reasons(features, probability, risk_category)

        return {
            "risk_category": risk_category,
            "churn_probability": probability,
            "churn_prediction": "Yes" if probability >= 0.5 else "No",
            "reasons": reasons,
            "ticket_count_last_30_days": int(features["ticket_count_30d"]),
            "model_version": str(self.metadata.get("model_version", "unknown")),
        }
