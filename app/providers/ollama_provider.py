import json
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.core.exceptions import (
    CircuitOpenError,
    ProviderError,
    ProviderTimeout,
    RateLimitError,
)
from app.core.logging import get_logger
from app.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError as _CBCircuitOpenError,
)
from app.resilience.error_isolation import (
    ErrorCategory,
    ErrorIsolator,
    categorize_http_status,
)
from app.providers.base import LLMProvider

logger = get_logger(__name__)


# Module-level resilience singletons
_circuit_breaker = CircuitBreaker(
    "ollama",
    failure_threshold=settings.circuit_failure_threshold,
    recovery_timeout_ms=settings.circuit_recovery_timeout_ms,
)
_error_isolator = ErrorIsolator(
    quarantine_duration_ms=settings.quarantine_duration_ms,
    quarantine_threshold=settings.quarantine_threshold,
)


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._timeout = timeout or settings.request_timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Client lifecycle — single shared connection pool
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init a long-lived AsyncClient for connection reuse."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Gracefully shut down the connection pool. Call on app shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("OllamaProvider connection pool closed")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(self, model: str, prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}

        # Circuit breaker check
        try:
            _circuit_breaker.allow_request()
        except _CBCircuitOpenError as exc:
            raise CircuitOpenError(f"Ollama circuit is open: {exc}") from exc

        # Check if service is quarantined
        if _error_isolator.is_quarantined("ollama"):
            qs = _error_isolator.get_quarantine_info("ollama")
            raise ProviderError(
                f"Ollama is quarantined ({qs.category.value if qs else 'unknown'}): "
                f"{qs.reason if qs else 'too many errors'}"
            )

        max_retries = settings.retry_max_attempts
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                client = self._get_client()
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()

                # Success — reset all failure tracking
                _circuit_breaker.record_success()
                _error_isolator.clear("ollama")

                return data.get("response", "")
            except httpx.TimeoutException as exc:
                last_exc = exc
                _circuit_breaker.record_failure()
                _error_isolator.record_error("ollama", ErrorCategory.TIMEOUT, str(exc))
                logger.warning("Ollama timeout on attempt %d/%d", attempt, max_retries)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                _error_isolator.record_error(
                    "ollama", categorize_http_status(status), exc.response.text
                )
                _circuit_breaker.record_failure()

                # Handle 429 rate limit specifically
                if status == 429:
                    raise RateLimitError(
                        f"Ollama rate limit exceeded (HTTP 429): {exc.response.text}"
                    ) from exc

                raise ProviderError(
                    f"Ollama returned HTTP {status}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                last_exc = exc
                _error_isolator.record_error(
                    "ollama", ErrorCategory.CONNECTION, str(exc)
                )
                _circuit_breaker.record_failure()
                logger.warning(
                    "Ollama request error on attempt %d/%d: %s",
                    attempt,
                    max_retries,
                    exc,
                )

        _circuit_breaker.record_failure()
        raise ProviderTimeout(
            f"Ollama did not respond after {max_retries} attempts"
        ) from last_exc

    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        payload = {"model": model, "prompt": prompt, "stream": True}

        # Circuit breaker check
        try:
            _circuit_breaker.allow_request()
        except _CBCircuitOpenError as exc:
            raise CircuitOpenError(f"Ollama circuit is open: {exc}") from exc

        if _error_isolator.is_quarantined("ollama"):
            qs = _error_isolator.get_quarantine_info("ollama")
            raise ProviderError(
                f"Ollama is quarantined ({qs.category.value if qs else 'unknown'}): "
                f"{qs.reason if qs else 'too many errors'}"
            )

        try:
            client = self._get_client()
            async with client.stream("POST", "/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        _error_isolator.record_error(
                            "ollama", ErrorCategory.PARSE, line[:100]
                        )
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

            # Stream completed successfully — reset all failure tracking
            _circuit_breaker.record_success()
            _error_isolator.clear("ollama")

        except httpx.TimeoutException as exc:
            _circuit_breaker.record_failure()
            _error_isolator.record_error("ollama", ErrorCategory.TIMEOUT, str(exc))
            raise ProviderTimeout("Ollama stream timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            _circuit_breaker.record_failure()
            _error_isolator.record_error(
                "ollama", categorize_http_status(status), exc.response.text
            )
            if status == 429:
                raise RateLimitError(
                    f"Ollama rate limit exceeded (HTTP 429): {exc.response.text}"
                ) from exc
            raise ProviderError(
                f"Ollama stream returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            _circuit_breaker.record_failure()
            _error_isolator.record_error("ollama", ErrorCategory.CONNECTION, str(exc))
            raise ProviderError(f"Ollama stream connection error: {exc}") from exc

    async def health(self) -> bool:
        try:
            client = self._get_client()
            resp = await client.get("/", timeout=5)
            is_healthy = resp.status_code < 500
            if is_healthy:
                _circuit_breaker.record_success()
                _error_isolator.clear("ollama")
            else:
                _circuit_breaker.record_failure()
                _error_isolator.record_error(
                    "ollama", categorize_http_status(resp.status_code), resp.text
                )
            return is_healthy
        except Exception as exc:
            _circuit_breaker.record_failure()
            _error_isolator.record_error("ollama", ErrorCategory.CONNECTION, str(exc))
            return False

    async def classify(self, model: str, prompt: str, timeout: int = 10) -> str:
        """Light-weight generate call for classifier use — bypasses circuit breaker.

        Uses ``temperature=0`` for deterministic routing decisions and a small
        ``num_predict`` cap so the model emits at most a single category word
        instead of a long explanation. This keeps classifier latency in the
        low hundreds of ms range on a local Ollama install.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 8,
                "stop": ["\n", "Prompt:", "Answer:"],
            },
        }
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(timeout, connect=5.0),
            ) as client:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as exc:
            raise ProviderError(f"Classifier call failed: {exc}") from exc

    def get_resilience_snapshot(self) -> dict:
        """Return circuit breaker + quarantine + error stats in one call."""
        quarantined = [
            {
                "service": qs.service_name,
                "category": qs.category.value,
                "reason": qs.reason,
                "error_count": qs.error_count,
            }
            for qs in _error_isolator.get_all_quarantined()
        ]
        return {
            "circuit_breaker": _circuit_breaker.get_stats(),
            "quarantined": quarantined,
            "error_stats": _error_isolator.get_stats("ollama"),
        }

    def reset_circuit_breaker(self) -> None:
        """Force-reset the circuit breaker to CLOSED state."""
        _circuit_breaker.reset()

    def release_quarantine(self, service: str = "ollama") -> bool:
        """Manually release a quarantined service. Returns True if it was quarantined."""
        return _error_isolator.release(service)
