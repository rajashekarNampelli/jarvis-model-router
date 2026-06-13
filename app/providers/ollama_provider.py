import json
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.core.exceptions import ProviderError, ProviderTimeout
from app.core.logging import get_logger
from app.providers.base import LLMProvider

logger = get_logger(__name__)

_MAX_RETRIES = 2


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._timeout = timeout or settings.request_timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(self, model: str, prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with self._client() as client:
                    resp = await client.post("/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "")
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("Ollama timeout on attempt %d/%d", attempt, _MAX_RETRIES)
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning("Ollama request error on attempt %d/%d: %s", attempt, _MAX_RETRIES, exc)

        raise ProviderTimeout(
            f"Ollama did not respond after {_MAX_RETRIES} attempts"
        ) from last_exc

    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        payload = {"model": model, "prompt": prompt, "stream": True}

        try:
            async with self._client() as client:
                async with client.stream("POST", "/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama stream timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama stream returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(f"Ollama stream connection error: {exc}") from exc

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=5) as client:
                resp = await client.get("/")
                return resp.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )
