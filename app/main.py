import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, health, models, resilience
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.metrics.prometheus import router as metrics_router
from app.middleware.request_logger import RequestLoggerMiddleware
from app.resilience.health_monitor import HealthMonitor
from app.providers import _provider

logger = logging.getLogger(__name__)

health_monitor: HealthMonitor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global health_monitor

    setup_logging(settings.log_level)

    health_monitor = HealthMonitor(
        check_interval_seconds=settings.health_check_interval_seconds,
        failure_threshold=settings.health_failure_threshold,
    )
    health_monitor.register("ollama", _provider.health)
    await health_monitor.start()

    logger.info("Jarvis started (health_monitor=on)")

    yield

    if health_monitor:
        await health_monitor.stop()

    await _provider.close()

    logger.info("Jarvis shutdown complete")


app = FastAPI(
    title="Jarvis Model Router",
    description="LLM routing with circuit breaker, error quarantine, and health monitoring",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggerMiddleware)

register_exception_handlers(app)

app.include_router(chat.router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(metrics_router)
app.include_router(resilience.router)
