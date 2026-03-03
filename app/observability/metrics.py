from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import time


BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


@dataclass
class MetricsState:
    started_at: float = field(default_factory=time)
    total_requests: int = 0
    active_requests: int = 0
    total_duration_seconds: float = 0.0
    request_durations: list[float] = field(default_factory=list)
    predictions: dict[str, int] = field(
        default_factory=lambda: {"Low": 0, "Medium": 0, "High": 0}
    )
    lock: Lock = field(default_factory=Lock)


_state = MetricsState()


def reset_metrics() -> None:
    global _state
    _state = MetricsState()


def begin_request() -> None:
    with _state.lock:
        _state.active_requests += 1


def end_request(duration_seconds: float) -> None:
    with _state.lock:
        _state.active_requests = max(0, _state.active_requests - 1)
        _state.total_requests += 1
        _state.total_duration_seconds += duration_seconds
        _state.request_durations.append(duration_seconds)


def record_prediction(risk_category: str) -> None:
    with _state.lock:
        if risk_category not in _state.predictions:
            _state.predictions[risk_category] = 0
        _state.predictions[risk_category] += 1


def _histogram_buckets() -> list[tuple[str, int]]:
    values = sorted(_state.request_durations)
    output: list[tuple[str, int]] = []

    for bucket in BUCKETS:
        count = sum(1 for value in values if value <= bucket)
        output.append((str(bucket), count))

    output.append(("+Inf", len(values)))
    return output


def render_metrics() -> str:
    with _state.lock:
        uptime = max(0.0, time() - _state.started_at)
        lines: list[str] = []

        lines.append("# HELP churn_requests_total Total HTTP requests served.")
        lines.append("# TYPE churn_requests_total counter")
        lines.append(f"churn_requests_total {_state.total_requests}")

        lines.append("# HELP churn_active_requests In-flight HTTP requests.")
        lines.append("# TYPE churn_active_requests gauge")
        lines.append(f"churn_active_requests {_state.active_requests}")

        lines.append(
            "# HELP churn_prediction_total Total churn predictions by risk category."
        )
        lines.append("# TYPE churn_prediction_total counter")
        for risk in ("Low", "Medium", "High"):
            lines.append(
                f'churn_prediction_total{{risk="{risk.lower()}"}} '
                f'{_state.predictions.get(risk, 0)}'
            )

        lines.append(
            "# HELP churn_request_duration_seconds Request duration for HTTP requests."
        )
        lines.append("# TYPE churn_request_duration_seconds histogram")
        for bucket, count in _histogram_buckets():
            lines.append(
                f'churn_request_duration_seconds_bucket{{le="{bucket}"}} {count}'
            )
        lines.append(
            f"churn_request_duration_seconds_count {_state.total_requests}"
        )
        lines.append(
            f"churn_request_duration_seconds_sum {_state.total_duration_seconds}"
        )

        lines.append("# HELP service_uptime_seconds Service uptime in seconds.")
        lines.append("# TYPE service_uptime_seconds gauge")
        lines.append(f"service_uptime_seconds {uptime}")
        return "\n".join(lines) + "\n"
