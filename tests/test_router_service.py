import pytest
from unittest.mock import AsyncMock, patch

from jarvis_model_router.schemas.chat import ChatRequest
from jarvis_model_router.services.router_service import RouterService
from jarvis_model_router.core.exceptions import ModelNotFound


@pytest.fixture
def router():
    return RouterService()


@pytest.mark.anyio
async def test_auto_routes_code_to_qwen(router: RouterService) -> None:
    with patch(
        "jarvis_model_router.routing.classifier._llm_classify", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.return_value = "qwen"
        req = ChatRequest(message="Write a Java unit test", model="auto")
        key, name = await router.route(req)
        assert key == "qwen"
        assert name == "qwen2.5-coder"


@pytest.mark.anyio
async def test_auto_routes_reasoning_to_deepseek(router: RouterService) -> None:
    with patch(
        "jarvis_model_router.routing.classifier._llm_classify", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.return_value = "deepseek"
        req = ChatRequest(message="Prove Pythagoras theorem", model="auto")
        key, name = await router.route(req)
        assert key == "deepseek"
        assert name == "deepseek-r1:8b"


@pytest.mark.anyio
async def test_auto_routes_general_to_llama(router: RouterService) -> None:
    with patch(
        "jarvis_model_router.routing.classifier._llm_classify", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.return_value = "llama"
        req = ChatRequest(message="What is the weather today?", model="auto")
        key, name = await router.route(req)
        assert key == "llama"
        assert name == "llama3"


@pytest.mark.anyio
async def test_explicit_model_key(router: RouterService) -> None:
    req = ChatRequest(message="anything", model="qwen")
    key, name = await router.route(req)
    assert key == "qwen"
    assert name == "qwen2.5-coder"


@pytest.mark.anyio
async def test_unknown_model_raises(router: RouterService) -> None:
    req = ChatRequest(message="hello", model="gpt5")
    with pytest.raises(ModelNotFound):
        await router.route(req)
