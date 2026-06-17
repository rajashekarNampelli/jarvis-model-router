from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter
from fastapi.responses import Response

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

request_count = Counter(
    "jarvis_request_total",
    "Total number of chat requests",
    ["model", "status"],
)

error_count = Counter(
    "jarvis_error_total",
    "Total number of failed chat requests",
    ["model", "error_type"],
)

request_latency = Histogram(
    "jarvis_request_latency_seconds",
    "Chat request latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

token_count = Counter(
    "jarvis_tokens_total",
    "Approximate total tokens processed",
    ["model"],
)

model_usage = Counter(
    "jarvis_model_usage_total",
    "How many times each model has been selected",
    ["model"],
)

# ---------------------------------------------------------------------------
# Helper functions called by InferenceService
# ---------------------------------------------------------------------------


def record_request(model: str, status: str, latency_seconds: float) -> None:
    request_count.labels(model=model, status=status).inc()
    request_latency.labels(model=model).observe(latency_seconds)
    model_usage.labels(model=model).inc()


def record_error(model: str, error_type: str) -> None:
    error_count.labels(model=model, error_type=error_type).inc()


def record_tokens(model: str, approx_tokens: int) -> None:
    token_count.labels(model=model).inc(approx_tokens)


# ---------------------------------------------------------------------------
# /metrics router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
