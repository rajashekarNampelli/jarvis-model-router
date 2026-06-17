import time
import uuid
from typing import AsyncIterator

from jarvis_model_router.core.exceptions import (
    JarvisBaseError,
    ProviderError,
    RateLimitError,
)
from jarvis_model_router.core.logging import get_logger, log_request
from jarvis_model_router.metrics.prometheus import (
    record_error,
    record_request,
    record_tokens,
)
from jarvis_model_router.providers import _provider
from jarvis_model_router.schemas.chat import ChatRequest
from jarvis_model_router.schemas.response import ChatResponse
from jarvis_model_router.services.router_service import RouterService

logger = get_logger(__name__)

_router = RouterService()


class InferenceService:
    async def generate(self, request: ChatRequest) -> ChatResponse:
        request_id = str(uuid.uuid4())
        model_key, ollama_model = await _router.route(request)

        start = time.monotonic()
        success = True
        response_text = ""

        try:
            response_text = await _provider.generate(ollama_model, request.message)
        except RateLimitError:
            success = False
            record_error(model_key, "rate_limit")
            raise
        except JarvisBaseError as exc:
            success = False
            record_error(model_key, type(exc).__name__)
            raise
        except Exception as exc:
            success = False
            record_error(model_key, "unexpected_error")
            raise ProviderError(f"Unexpected provider error: {exc}") from exc
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            latency_seconds = latency_ms / 1000.0
            status = "success" if success else "error"
            record_request(model_key, status, latency_seconds)

            approx_tokens = len(request.message.split()) + len(response_text.split())
            record_tokens(model_key, approx_tokens)

            log_request(
                logger,
                request_id=request_id,
                model=model_key,
                latency_ms=latency_ms,
                success=success,
                prompt_length=len(request.message),
                prompt_tokens_approx=len(request.message.split()),
            )

        return ChatResponse(
            request_id=request_id,
            selected_model=model_key,
            response=response_text,
            latency_ms=latency_ms,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        request_id = str(uuid.uuid4())
        model_key, ollama_model = await _router.route(request)

        start = time.monotonic()
        success = True
        token_buffer: list[str] = []

        try:
            async for token in _provider.stream(ollama_model, request.message):
                token_buffer.append(token)
                yield token
        except RateLimitError:
            success = False
            record_error(model_key, "rate_limit")
            raise
        except JarvisBaseError as exc:
            success = False
            record_error(model_key, type(exc).__name__)
            raise
        except Exception as exc:
            success = False
            record_error(model_key, "unexpected_error")
            raise ProviderError(f"Unexpected provider error: {exc}") from exc
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            status = "success" if success else "error"
            record_request(model_key, status, latency_ms / 1000.0)

            full_response = "".join(token_buffer)
            approx_tokens = len(request.message.split()) + len(full_response.split())
            record_tokens(model_key, approx_tokens)

            log_request(
                logger,
                request_id=request_id,
                model=model_key,
                latency_ms=latency_ms,
                success=success,
                prompt_length=len(request.message),
                prompt_tokens_approx=len(request.message.split()),
            )
