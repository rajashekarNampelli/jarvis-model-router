import json
import logging
import sys
from typing import Any


# Keys that our JSONFormatter should include from the extra dict.
# Everything else in record.__dict__ is internal Python logging noise.
_LOG_STRUCT_KEYS = frozenset({
    "request_id", "model", "latency_ms", "success",
    "prompt_length", "prompt_tokens_approx",
    "method", "path", "status_code",
    "error", "detail",
})


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Pull structured keys from record.__dict__ (set via `extra={}`)
        for key in _LOG_STRUCT_KEYS:
            if key in record.__dict__:
                log_obj[key] = record.__dict__[key]
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(log_level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    model: str,
    latency_ms: int,
    success: bool,
    prompt_length: int,
    prompt_tokens_approx: int,
) -> None:
    """Emit a structured per-request log. Never logs the prompt itself."""
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "model": model,
            "latency_ms": latency_ms,
            "success": success,
            "prompt_length": prompt_length,
            "prompt_tokens_approx": prompt_tokens_approx,
        },
    )
