import pytest
from unittest.mock import AsyncMock, patch

from jarvis_model_router.schemas.chat import ChatRequest
from jarvis_model_router.services.inference_service import InferenceService
from jarvis_model_router.core.exceptions import ProviderError


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
