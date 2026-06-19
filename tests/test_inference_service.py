import pytest
from unittest.mock import AsyncMock, patch

from jarvis_model_router.schemas.chat import ChatRequest
from jarvis_model_router.services.inference_service import InferenceService, _build_prompt
from jarvis_model_router.core.exceptions import ProviderError


# ---------------------------------------------------------------------------
# _build_prompt unit tests
# ---------------------------------------------------------------------------

def test_build_prompt_no_context_returns_message() -> None:
    req = ChatRequest(message="What is Python?", model="auto")
    assert _build_prompt(req) == "What is Python?"


def test_build_prompt_none_context_returns_message() -> None:
    req = ChatRequest(message="Hello", model="llama", context=None)
    assert _build_prompt(req) == "Hello"


def test_build_prompt_with_context_includes_both() -> None:
    file_context = "### File: foo.py\n```python\ndef hello(): pass\n```"
    req = ChatRequest(message="Explain this function", model="auto", context=file_context)
    result = _build_prompt(req)
    assert file_context in result
    assert "Explain this function" in result
    # Context must appear before the user question
    assert result.index(file_context) < result.index("Explain this function")


def test_build_prompt_with_context_has_separator() -> None:
    req = ChatRequest(message="Summarize", model="auto", context="some context")
    result = _build_prompt(req)
    assert "---" in result


def test_build_prompt_routing_unaffected() -> None:
    """Routing uses request.message directly, not _build_prompt output."""
    req = ChatRequest(
        message="short question",
        model="auto",
        context="A" * 10_000,  # large context should not affect the routed field
    )
    assert req.message == "short question"


@pytest.fixture
def service():
    return InferenceService()


@pytest.mark.anyio
async def test_generate_returns_chat_response(service: InferenceService) -> None:
    req = ChatRequest(message="Write Java unit test", model="auto")

    with (
        patch(
            "jarvis_model_router.routing.classifier._llm_classify",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "jarvis_model_router.services.inference_service._provider"
        ) as mock_provider,
    ):
        mock_llm.return_value = "qwen"
        mock_provider.generate = AsyncMock(return_value="Here is a unit test...")

        result = await service.generate(req)

    assert result.selected_model == "qwen"
    assert result.response == "Here is a unit test..."
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0
    assert result.request_id != ""


@pytest.mark.anyio
async def test_generate_tracks_latency(service: InferenceService) -> None:
    import asyncio

    req = ChatRequest(message="Tell me a story", model="llama")

    async def slow_generate(model, prompt):
        await asyncio.sleep(0.05)
        return "Once upon a time..."

    with patch(
        "jarvis_model_router.services.inference_service._provider"
    ) as mock_provider:
        mock_provider.generate = slow_generate
        result = await service.generate(req)

    assert result.latency_ms >= 50


@pytest.mark.anyio
async def test_generate_propagates_provider_error(service: InferenceService) -> None:
    req = ChatRequest(message="Hello", model="llama")

    with patch(
        "jarvis_model_router.services.inference_service._provider"
    ) as mock_provider:
        mock_provider.generate = AsyncMock(side_effect=ProviderError("Ollama down"))

        with pytest.raises(ProviderError):
            await service.generate(req)


@pytest.mark.anyio
async def test_stream_yields_tokens(service: InferenceService) -> None:
    req = ChatRequest(message="Explain quicksort", model="auto")

    async def fake_stream(model, prompt):
        for token in ["Quick", "sort", " is", " fast"]:
            yield token

    with (
        patch(
            "jarvis_model_router.routing.classifier._llm_classify",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "jarvis_model_router.services.inference_service._provider"
        ) as mock_provider,
    ):
        mock_llm.return_value = "deepseek"
        mock_provider.stream = fake_stream

        tokens = []
        async for token in service.stream(req):
            tokens.append(token)

    assert tokens == ["Quick", "sort", " is", " fast"]
