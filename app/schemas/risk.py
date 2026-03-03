from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TicketRecord(BaseModel):
    opened_at: datetime = Field(description="Ticket creation timestamp in ISO format.")
    category: str = Field(default="general", description="Ticket category.")
    is_complaint: bool = Field(
        default=False, description="Whether ticket is a complaint."
    )
    sentiment_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional ticket sentiment score.",
    )


class CustomerProfile(BaseModel):
    customer_id: str = Field(min_length=1, description="Unique customer identifier.")
    contract_type: str = Field(description="Contract type, e.g., Month-to-Month.")
    monthly_charges_current: float = Field(ge=0, description="Current monthly charges.")
    monthly_charges_previous: float = Field(
        ge=0, description="Previous monthly charges."
    )
    tenure_months: int = Field(
        default=12, ge=0, description="Customer tenure in months."
    )
    total_charges: float | None = Field(
        default=None,
        ge=0,
        description="Optional total charges. If missing, inferred from monthly charges and tenure.",
    )
    internet_service: str = Field(
        default="Fiber optic", description="Type of internet service."
    )
    payment_method: str = Field(
        default="Electronic check", description="Customer payment method."
    )


class RiskRequest(BaseModel):
    customer: CustomerProfile
    tickets: list[TicketRecord] = Field(default_factory=list)


class RiskResponse(BaseModel):
    risk_category: Literal["Low", "Medium", "High"]
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: Literal["Yes", "No"]
    reasons: list[str]
    ticket_count_last_30_days: int
    model_version: str
    evaluated_at: datetime
