from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from jarvis_model_router.schemas.chat import ChatRequest
from jarvis_model_router.schemas.response import ChatResponse
from jarvis_model_router.services.inference_service import InferenceService

router = APIRouter()
_inference = InferenceService()


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await _inference.generate(request)


@router.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def sse_generator():
        async for token in _inference.stream(request):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
