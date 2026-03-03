from datetime import datetime, timedelta, timezone

from app.schemas.risk import RiskRequest


def _is_month_to_month(contract_type: str) -> bool:
    normalized = contract_type.strip().lower().replace(" ", "").replace("-", "")
    return normalized == "monthtomonth"


def _ticket_count_last_n_days(payload: RiskRequest, now: datetime, days: int) -> int:
    cutoff = now - timedelta(days=days)
    count = 0
    for ticket in payload.tickets:
        opened_at = ticket.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        if opened_at >= cutoff:
            count += 1
    return count


def _has_complaint_ticket(payload: RiskRequest) -> bool:
    return any(ticket.is_complaint for ticket in payload.tickets)


def evaluate_risk(payload: RiskRequest, now: datetime | None = None) -> dict[str, object]:
    """Apply pure business rules for churn risk classification."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ticket_count_last_30 = _ticket_count_last_n_days(payload, now, days=30)

    high_rules: list[str] = []
    medium_rules: list[str] = []

    if ticket_count_last_30 > 5:
        high_rules.append("More than 5 tickets were raised in the last 30 days.")

    if (
        payload.customer.monthly_charges_current
        > payload.customer.monthly_charges_previous
        and ticket_count_last_30 >= 3
    ):
        medium_rules.append(
            "Monthly charges increased and at least 3 tickets were raised."
        )

    if _is_month_to_month(payload.customer.contract_type) and _has_complaint_ticket(
        payload
    ):
        high_rules.append("Month-to-Month contract with complaint ticket detected.")

    if high_rules:
        risk = "High"
        reasons = high_rules + medium_rules
    elif medium_rules:
        risk = "Medium"
        reasons = medium_rules
    else:
        risk = "Low"
        reasons = ["No high-risk or medium-risk churn rule matched."]

    return {
        "risk_category": risk,
        "reasons": reasons,
        "ticket_count_last_30_days": ticket_count_last_30,
    }
