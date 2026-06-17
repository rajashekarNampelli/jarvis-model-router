from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's prompt")
    model: str = Field(default="auto", description="Model key or 'auto' for routing")
