from fastapi import APIRouter

from app.providers import _provider

router = APIRouter()


@router.get("/v1/resilience")
async def resilience_status() -> dict:
    """Circuit breaker, quarantine, error stats, and health monitor status."""
    result = _provider.get_resilience_snapshot()

    try:
        from app.main import health_monitor

        result["health_monitor"] = (
            health_monitor.get_summary() if health_monitor else {}
        )
    except Exception:
        result["health_monitor"] = {}

    return result


@router.post("/v1/resilience/circuit-breaker/reset")
async def reset_circuit_breaker() -> dict:
    """Force-reset the Ollama circuit breaker to CLOSED state."""
    _provider.reset_circuit_breaker()
    return {"status": "reset", "circuit_state": "closed"}


@router.post("/v1/resilience/quarantine/release")
async def release_quarantine(service: str = "ollama") -> dict:
    """Manually release a quarantined service."""
    released = _provider.release_quarantine(service)
    return {"service": service, "released": released}
