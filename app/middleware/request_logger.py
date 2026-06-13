import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger("jarvis.access")


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.monotonic()
        response = await call_next(request)
        latency_ms = int((time.monotonic() - start) * 1000)

        record = logger.makeRecord(
            logger.name,
            20,  # INFO
            fn="",
            lno=0,
            msg="http_request",
            args=(),
            exc_info=None,
        )
        record.extra = {  # type: ignore[attr-defined]
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
        logger.handle(record)

        response.headers["X-Request-ID"] = request_id
        return response
