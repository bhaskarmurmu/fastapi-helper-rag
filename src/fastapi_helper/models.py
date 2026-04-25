from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    id: int
    text: str
    url: str
    type: Literal["docs", "issue"]
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    cache_hit: bool
    latency_ms: int
    request_id: str


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    qdrant: bool
    postgres: bool
    timestamp: datetime
