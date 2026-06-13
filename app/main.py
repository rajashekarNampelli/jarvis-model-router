from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, health, models
from app.core.config import settings
from app.core.exceptions import JarvisBaseError, register_exception_handlers
from app.core.logging import setup_logging
from app.metrics.prometheus import router as metrics_router
from app.middleware.request_logger import RequestLoggerMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    yield


app = FastAPI(
    title="Jarvis Model Router",
    description="Phase 1: deterministic LLM routing over Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggerMiddleware)

register_exception_handlers(app)

app.include_router(chat.router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(metrics_router)
