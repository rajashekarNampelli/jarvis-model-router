from typing import Literal
from pydantic import BaseModel


class ChatResponse(BaseModel):
    request_id: str
    selected_model: str
    response: str
    latency_ms: int


class MemoryInfo(BaseModel):
    used_mb: float
    total_mb: float
    percent: float


class HealthResponse(BaseModel):
    status: Literal["UP", "DOWN"]
    ollama: Literal["UP", "DOWN"]
    memory: MemoryInfo
    gpu: str
