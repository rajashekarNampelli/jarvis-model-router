from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's prompt")
    model: str = Field(default="auto", description="Model key or 'auto' for routing")
    context: str | None = Field(
        default=None,
        description="Optional workspace file context injected before the message at inference time. "
                    "Routing always uses the raw 'message' field so auto-routing is unaffected.",
    )
