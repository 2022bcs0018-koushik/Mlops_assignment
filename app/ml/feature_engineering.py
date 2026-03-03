from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable


TICKET_CATEGORY_BUCKETS = ("complaint", "billing", "technical", "general")


@dataclass
class TicketObservation:
    opened_at: datetime
    category: str = "general"
    is_complaint: bool = False
    sentiment_score: float | None = None


@dataclass
class CustomerObservation:
    customer_id: str
    contract_type: str
    monthly_charges_current: float
    monthly_charges_previous: float
    tenure_months: int = 12
    total_charges: float | None = None
    internet_service: str = "Fiber optic"
    payment_method: str = "Electronic check"


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalized_category(category: str) -> str:
    normalized = (category or "general").strip().lower()
    if normalized in TICKET_CATEGORY_BUCKETS:
        return normalized
    return "general"


def build_feature_vector(
    customer: CustomerObservation,
    tickets: Iterable[TicketObservation],
    reference_time: datetime | None = None,
) -> dict[str, float | int | str]:
    now = _as_utc(reference_time or datetime.now(timezone.utc))

    ticket_list = sorted(
        ((_as_utc(ticket.opened_at), ticket) for ticket in tickets),
        key=lambda item: item[0],
    )
    ticket_count_7d = 0
    ticket_count_30d = 0
    ticket_count_90d = 0
    sentiment_scores: list[float] = []
    category_counts = {name: 0 for name in TICKET_CATEGORY_BUCKETS}
    complaint_ticket_count = 0

    for opened_at, ticket in ticket_list:
        age_days = (now - opened_at).total_seconds() / 86400
        category = _normalized_category(ticket.category)

        if age_days <= 90:
            ticket_count_90d += 1
            category_counts[category] += 1
            if category == "complaint" or ticket.is_complaint:
                complaint_ticket_count += 1
            if ticket.sentiment_score is not None:
                sentiment_scores.append(float(ticket.sentiment_score))

            if age_days <= 30:
                ticket_count_30d += 1
            if age_days <= 7:
                ticket_count_7d += 1

    avg_ticket_sentiment = mean(sentiment_scores) if sentiment_scores else 0.0

    if len(ticket_list) >= 2:
        diffs = [
            (ticket_list[i][0] - ticket_list[i - 1][0]).total_seconds() / 86400
            for i in range(1, len(ticket_list))
        ]
        avg_days_between_tickets = mean(diffs)
    else:
        avg_days_between_tickets = 90.0

    monthly_charge_change = (
        float(customer.monthly_charges_current) - float(customer.monthly_charges_previous)
    )
    tenure_months = int(customer.tenure_months)
    inferred_total = float(customer.monthly_charges_current) * max(tenure_months, 1)
    total_charges = (
        float(customer.total_charges)
        if customer.total_charges is not None
        else inferred_total
    )

    return {
        "ticket_count_7d": ticket_count_7d,
        "ticket_count_30d": ticket_count_30d,
        "ticket_count_90d": ticket_count_90d,
        "avg_ticket_sentiment": float(avg_ticket_sentiment),
        "complaint_ticket_count": complaint_ticket_count,
        "billing_ticket_count": category_counts["billing"],
        "technical_ticket_count": category_counts["technical"],
        "general_ticket_count": category_counts["general"],
        "avg_days_between_tickets": float(avg_days_between_tickets),
        "monthly_charge_change": float(monthly_charge_change),
        "monthly_charges_current": float(customer.monthly_charges_current),
        "tenure_months": tenure_months,
        "total_charges": float(total_charges),
        "contract_type": customer.contract_type,
        "internet_service": customer.internet_service,
        "payment_method": customer.payment_method,
    }
