from typing import TypedDict


class ModelEntry(TypedDict):
    provider: str
    name: str
    timeout_seconds: int


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "llama": {
        "provider": "ollama",
        "name": "llama3",
        "timeout_seconds": 120,
    },
    "qwen": {
        "provider": "ollama",
        "name": "qwen2.5-coder",
        "timeout_seconds": 120,
    },
    "deepseek": {
        "provider": "ollama",
        "name": "deepseek-r1:8b",
        "timeout_seconds": 300,  # Reasoning model takes longer
    },
}

# Default model key for "auto" routing when classification fails or prompt is empty.
DEFAULT_MODEL_KEY: str = "llama"
