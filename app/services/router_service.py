"""
RouterService — the brain of the routing layer.

Resolves the model key from a request:
  - "auto"         → classifier decides
  - explicit key   → validated against registry
  - explicit Ollama model name → passed through as-is

Returns a (model_key, ollama_model_name) tuple.
"""

from app.core.exceptions import ModelNotFound
from app.routing.classifier import classify
from app.routing.rules import MODEL_REGISTRY
from app.schemas.chat import ChatRequest


class RouterService:
    def route(self, request: ChatRequest) -> tuple[str, str]:
        """
        Return (model_key, ollama_model_name).

        Raises ModelNotFound if an explicit key is not in the registry.
        """
        if request.model == "auto":
            model_key = classify(request.message)
        else:
            model_key = request.model

        if model_key not in MODEL_REGISTRY:
            raise ModelNotFound(
                f"Model '{model_key}' not found. "
                f"Available: {', '.join(MODEL_REGISTRY.keys())}"
            )

        entry = MODEL_REGISTRY[model_key]
        return model_key, entry["name"]
