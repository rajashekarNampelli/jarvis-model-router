from fastapi import APIRouter
from jarvis_model_router.routing.rules import MODEL_REGISTRY

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> dict:
    return {
        "models": [
            {"key": key, "provider": entry["provider"], "name": entry["name"]}
            for key, entry in MODEL_REGISTRY.items()
        ]
    }
