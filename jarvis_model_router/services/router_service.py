"""
RouterService — the brain of the routing layer.

Resolves the model key from a request:
  - "auto"         → LLM classifier decides (falls back to default model on failure)
  - explicit key   → validated against registry

Returns a (model_key, ollama_model_name) tuple.
"""

from jarvis_model_router.core.exceptions import ModelNotFound
from jarvis_model_router.routing.classifier import classify
from jarvis_model_router.routing.rules import MODEL_REGISTRY
from jarvis_model_router.schemas.chat import ChatRequest


class RouterService:
    async def route(self, request: ChatRequest) -> tuple[str, str]:
        """
        Return (model_key, ollama_model_name).

        Raises ModelNotFound if an explicit key is not in the registry.
        """
        if request.model == "auto":
            model_key = await classify(request.message)
        else:
            model_key = request.model

        if model_key not in MODEL_REGISTRY:
            raise ModelNotFound(
                f"Model '{model_key}' not found. "
                f"Available: {', '.join(MODEL_REGISTRY.keys())}"
            )

        entry = MODEL_REGISTRY[model_key]
        return model_key, entry["name"]
