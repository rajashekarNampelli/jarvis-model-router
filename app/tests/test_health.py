import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.anyio
async def test_health_ollama_up(client) -> None:
    with patch("app.api.health._provider") as mock_provider:
        mock_provider.health = AsyncMock(return_value=True)

        resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "UP"
    assert body["ollama"] == "UP"
    assert "memory" in body
    assert "gpu" in body


@pytest.mark.anyio
async def test_health_ollama_down(client) -> None:
    with patch("app.api.health._provider") as mock_provider:
        mock_provider.health = AsyncMock(return_value=False)

        resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "UP"
    assert body["ollama"] == "DOWN"


@pytest.mark.anyio
async def test_health_memory_fields(client) -> None:
    with patch("app.api.health._provider") as mock_provider:
        mock_provider.health = AsyncMock(return_value=True)

        resp = await client.get("/health")

    body = resp.json()
    mem = body["memory"]
    assert "used_mb" in mem
    assert "total_mb" in mem
    assert "percent" in mem
    assert mem["total_mb"] > 0


@pytest.mark.anyio
async def test_metrics_endpoint(client) -> None:
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "jarvis_request_total" in resp.text or "jarvis" in resp.text
