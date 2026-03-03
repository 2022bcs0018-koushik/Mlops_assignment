NUMERIC_FEATURES = [
    "ticket_count_7d",
    "ticket_count_30d",
    "ticket_count_90d",
    "avg_ticket_sentiment",
    "complaint_ticket_count",
    "billing_ticket_count",
    "technical_ticket_count",
    "general_ticket_count",
    "avg_days_between_tickets",
    "monthly_charge_change",
    "monthly_charges_current",
    "tenure_months",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]

MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_RISK_THRESHOLDS = {
    "medium": 0.40,
    "high": 0.70,
}
