import pytest
from unittest.mock import AsyncMock, patch

from app.routing.classifier import classify


# ── LLM classifier tests (mock the _llm_classify function directly) ─────────


@pytest.mark.anyio
async def test_llm_classify_code() -> None:
    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "qwen"
        result = await classify("Write a Rust web server")
        assert result == "qwen"


@pytest.mark.anyio
async def test_llm_classify_reasoning() -> None:
    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "deepseek"
        result = await classify("Prove Pythagoras theorem")
        assert result == "deepseek"


@pytest.mark.anyio
async def test_llm_classify_general() -> None:
    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "llama"
        result = await classify("Tell me a story about dragons")
        assert result == "llama"


@pytest.mark.anyio
async def test_llm_classify_none_defaults_to_llama() -> None:
    """If LLM returns None (unparseable or failed), default to general."""
    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        result = await classify("Write a Python function")
        assert result == "llama"  # safe default


@pytest.mark.anyio
async def test_llm_classify_failure_defaults_to_llama() -> None:
    """If LLM throws, default to general."""
    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("Ollama down")
        result = await classify("Solve this differential equation")
        assert result == "llama"  # safe default


@pytest.mark.anyio
async def test_llm_classify_empty_defaults_to_llama() -> None:
    result = await classify("")
    assert result == "llama"


@pytest.mark.anyio
async def test_llm_classify_jdk_routes_code() -> None:
    """The JDK prompt that misrouted with keyword classifier -- LLM catches it."""
    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "qwen"
        result = await classify("What are the key differences between JDK 8 and JDK 21?")
        assert result == "qwen"


# ── Cache tests ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_classify_caches_result() -> None:
    """Second call with same prompt should use cache (no second LLM call)."""
    from app.routing.classifier import _cache
    _cache._store.clear()

    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "qwen"
        result1 = await classify("Build a Spring Boot app")
        result2 = await classify("Build a Spring Boot app")
        assert result1 == "qwen"
        assert result2 == "qwen"
        # LLM should only be called once (cache hit on second call)
        assert mock_llm.call_count == 1


@pytest.mark.anyio
async def test_classify_caches_llama_on_failure() -> None:
    """Failed classification should cache the fallback so we don't retry."""
    from app.routing.classifier import _cache
    _cache._store.clear()

    with patch("app.routing.classifier._llm_classify", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        result1 = await classify("Something ambiguous")
        result2 = await classify("Something ambiguous")
        assert result1 == "llama"
        assert result2 == "llama"
        assert mock_llm.call_count == 1  # cached the fallback, no retry
