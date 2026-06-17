import asyncio
import asyncio.subprocess

import psutil
from fastapi import APIRouter

from app.providers import _provider
from app.schemas.response import HealthResponse, MemoryInfo

router = APIRouter()


def _memory_info() -> MemoryInfo:
    vm = psutil.virtual_memory()
    return MemoryInfo(
        used_mb=round(vm.used / (1024**2), 1),
        total_mb=round(vm.total / (1024**2), 1),
        percent=vm.percent,
    )


async def _gpu_info() -> str:
    """Non-blocking nvidia-smi query; returns 'unavailable' if not present."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
        output = stdout.decode().strip()
        return output if output else "unavailable"
    except Exception:
        return "unavailable"


def _health_monitor_summary() -> dict:
    try:
        from app.main import health_monitor

        return health_monitor.get_summary() if health_monitor else {}
    except Exception:
        return {}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_up = await _provider.health()
    return HealthResponse(
        status="UP",
        ollama="UP" if ollama_up else "DOWN",
        memory=_memory_info(),
        gpu=await _gpu_info(),
    )


@router.get("/health/detailed")
async def health_detailed() -> dict:
    """Extended health endpoint with resilience and monitoring details."""
    ollama_up = await _provider.health()
    return {
        "status": "UP",
        "ollama": "UP" if ollama_up else "DOWN",
        "memory": _memory_info().model_dump(),
        "gpu": await _gpu_info(),
        "resilience": _provider.get_resilience_snapshot(),
        "health_monitor": _health_monitor_summary(),
    }


@router.get("/livez")
async def livez() -> dict:
    """Liveness probe — always 200 if the process is running."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict:
    """Readiness probe — 503 if circuit is open or Ollama is quarantined."""
    from fastapi.responses import JSONResponse

    snapshot = _provider.get_resilience_snapshot()
    circuit_state = snapshot.get("circuit_breaker", {}).get("state", "unknown")
    quarantined = snapshot.get("quarantined", [])

    if circuit_state == "open" or quarantined:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "circuit_state": circuit_state,
                "quarantined": quarantined,
            },
        )

    return {"status": "ready", "circuit_state": circuit_state}
