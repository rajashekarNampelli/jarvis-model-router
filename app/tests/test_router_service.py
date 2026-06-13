import pytest
from app.schemas.chat import ChatRequest
from app.services.router_service import RouterService
from app.core.exceptions import ModelNotFound


@pytest.fixture
def router():
    return RouterService()


def test_auto_routes_code_to_qwen(router: RouterService) -> None:
    req = ChatRequest(message="Write a Java unit test", model="auto")
    key, name = router.route(req)
    assert key == "qwen"
    assert name == "qwen2.5-coder"


def test_auto_routes_reasoning_to_deepseek(router: RouterService) -> None:
    req = ChatRequest(message="Prove Pythagoras theorem", model="auto")
    key, name = router.route(req)
    assert key == "deepseek"
    assert name == "deepseek-r1:8b"


def test_auto_routes_general_to_llama(router: RouterService) -> None:
    req = ChatRequest(message="What is the weather today?", model="auto")
    key, name = router.route(req)
    assert key == "llama"
    assert name == "llama3"


def test_explicit_model_key(router: RouterService) -> None:
    req = ChatRequest(message="anything", model="qwen")
    key, name = router.route(req)
    assert key == "qwen"
    assert name == "qwen2.5-coder"


def test_unknown_model_raises(router: RouterService) -> None:
    req = ChatRequest(message="hello", model="gpt5")
    with pytest.raises(ModelNotFound):
        router.route(req)
