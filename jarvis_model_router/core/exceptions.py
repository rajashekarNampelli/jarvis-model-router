import logging

from fastapi import Request
from fastapi.responses import JSONResponse


class JarvisBaseError(Exception):
    """Base class for all domain errors."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProviderError(JarvisBaseError):
    status_code = 502
    error_code = "provider_error"


class ProviderTimeout(JarvisBaseError):
    status_code = 504
    error_code = "provider_timeout"


class ModelNotFound(JarvisBaseError):
    status_code = 404
    error_code = "model_not_found"


class RateLimitError(JarvisBaseError):
    status_code = 429
    error_code = "rate_limit"


class CircuitOpenError(JarvisBaseError):
    status_code = 503
    error_code = "circuit_open"


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------


async def jarvis_error_handler(request: Request, exc: JarvisBaseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "detail": exc.message},
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.getLogger("jarvis.errors").exception(
        "Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


def register_exception_handlers(app) -> None:  # noqa: ANN001
    app.add_exception_handler(JarvisBaseError, jarvis_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
