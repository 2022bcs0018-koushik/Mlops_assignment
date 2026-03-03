import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from app.api.routes import router as api_router
from app.core.logging_config import configure_logging
from app.observability.metrics import begin_request, end_request, render_metrics

configure_logging()
logger = logging.getLogger("churn_service")

app = FastAPI(
    title="ML-Based Churn Risk Service",
    version="2.0.0",
    description=(
        "Task 2 microservice that predicts churn risk using a trained machine "
        "learning model with ticket-derived features."
    ),
)
app.include_router(api_router)


@app.get("/metrics", tags=["Observability"])
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        render_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    start = perf_counter()
    begin_request()
    request_id = request.headers.get("X-Request-ID", str(uuid4()))

    try:
        response = await call_next(request)
        return response
    finally:
        duration_seconds = perf_counter() - start
        end_request(duration_seconds)
        logger.info(
            "request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_seconds * 1000,
        )
