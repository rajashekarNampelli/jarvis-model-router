import psutil
from fastapi import APIRouter

from app.providers import _provider
from app.schemas.response import HealthResponse, MemoryInfo

router = APIRouter()



def _memory_info() -> MemoryInfo:
    vm = psutil.virtual_memory()
    return MemoryInfo(
        used_mb=round(vm.used / (1024 ** 2), 1),
        total_mb=round(vm.total / (1024 ** 2), 1),
        percent=vm.percent,
    )


def _gpu_info() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "unavailable"
    except Exception:
        return "unavailable"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_up = await _provider.health()
    return HealthResponse(
        status="UP",
        ollama="UP" if ollama_up else "DOWN",
        memory=_memory_info(),
        gpu=_gpu_info(),
    )
