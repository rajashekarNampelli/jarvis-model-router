from typing import TypedDict


class ModelEntry(TypedDict):
    provider: str
    name: str


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "llama": {
        "provider": "ollama",
        "name": "llama3",
    },
    "qwen": {
        "provider": "ollama",
        "name": "qwen2.5-coder",
    },
    "deepseek": {
        "provider": "ollama",
        "name": "deepseek-r1:8b",
    },
}
