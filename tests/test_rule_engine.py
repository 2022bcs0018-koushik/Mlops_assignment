from datetime import datetime, timedelta, timezone

from app.schemas.risk import CustomerProfile, RiskRequest, TicketRecord
from app.services.rule_engine import evaluate_risk


def _payload(
    *,
    contract_type: str = "Two year",
    current: float = 75.0,
    previous: float = 75.0,
    ticket_days: list[int] | None = None,
    complaint: bool = False,
) -> RiskRequest:
    now = datetime.now(timezone.utc)
    days = ticket_days or []
    tickets = [
        TicketRecord(
            opened_at=now - timedelta(days=day),
            category="billing",
            is_complaint=complaint,
        )
        for day in days
    ]
    return RiskRequest(
        customer=CustomerProfile(
            customer_id="C-101",
            contract_type=contract_type,
            monthly_charges_current=current,
            monthly_charges_previous=previous,
        ),
        tickets=tickets,
    )


def test_high_risk_if_more_than_five_tickets_in_last_30_days():
    payload = _payload(ticket_days=[1, 2, 3, 4, 5, 6])
    result = evaluate_risk(payload)
    assert result["risk_category"] == "High"
    assert result["ticket_count_last_30_days"] == 6


def test_medium_risk_if_charges_increase_and_three_tickets():
    payload = _payload(current=90.0, previous=75.0, ticket_days=[2, 4, 10])
    result = evaluate_risk(payload)
    assert result["risk_category"] == "Medium"


def test_high_risk_if_month_to_month_and_complaint_ticket_exists():
    payload = _payload(
        contract_type="Month-to-Month",
        ticket_days=[40, 60],
        complaint=True,
    )
    result = evaluate_risk(payload)
    assert result["risk_category"] == "High"


def test_low_risk_if_no_rule_matches():
    payload = _payload(ticket_days=[45], current=70.0, previous=70.0)
    result = evaluate_risk(payload)
    assert result["risk_category"] == "Low"


def test_high_risk_overrides_medium_risk():
    payload = _payload(
        contract_type="Month-to-Month",
        current=95.0,
        previous=60.0,
        ticket_days=[1, 2, 3, 4, 5, 6],
        complaint=True,
    )
    result = evaluate_risk(payload)
    assert result["risk_category"] == "High"
    assert any("Monthly charges increased" in reason for reason in result["reasons"])
