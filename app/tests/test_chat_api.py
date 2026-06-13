import pytest
from unittest.mock import AsyncMock, patch

from app.schemas.response import ChatResponse


@pytest.mark.anyio
async def test_chat_endpoint_success(client) -> None:
    mock_response = ChatResponse(
        request_id="test-id",
        selected_model="qwen",
        response="Here is a Java unit test.",
        latency_ms=100,
    )

    with patch("app.api.chat._inference") as mock_inference:
        mock_inference.generate = AsyncMock(return_value=mock_response)

        resp = await client.post(
            "/v1/chat",
            json={"message": "Write Java unit test", "model": "auto", "stream": False},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_model"] == "qwen"
    assert body["response"] == "Here is a Java unit test."
    assert "request_id" in body
    assert "latency_ms" in body


@pytest.mark.anyio
async def test_chat_endpoint_empty_message(client) -> None:
    resp = await client.post(
        "/v1/chat",
        json={"message": "", "model": "auto"},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_chat_endpoint_unknown_model(client) -> None:
    resp = await client.post(
        "/v1/chat",
        json={"message": "hello", "model": "nonexistent"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "model_not_found"


@pytest.mark.anyio
async def test_chat_stream_endpoint(client) -> None:
    async def fake_stream(request):
        for token in ["Hello", " world"]:
            yield token

    with patch("app.api.chat._inference") as mock_inference:
        mock_inference.stream = fake_stream

        resp = await client.post(
            "/v1/chat/stream",
            json={"message": "Say hello", "model": "auto", "stream": True},
        )

    assert resp.status_code == 200
    assert "Hello" in resp.text


@pytest.mark.anyio
async def test_models_endpoint(client) -> None:
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    keys = [m["key"] for m in body["models"]]
    assert "llama" in keys
    assert "qwen" in keys
    assert "deepseek" in keys
