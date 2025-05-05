import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class MessageInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID cannot be empty")
    conversation_id: str = Field(..., min_length=1, description="Conversation ID cannot be empty")
    type: str = Field(..., pattern="^(human|ai)$", description="Must be 'human' or 'ai'")
    text: str = Field(..., min_length=1, description="Message text cannot be empty.")
    timestamp: str | None = Field(None, description="UTC timestamp (optional)")

class SearchRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    query: str = Field(..., description="Search query")

class RememberRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    content: str = Field(..., description="Content to remember")

class MemoryNode(BaseModel):
    """Hierarchical memory node with importance scoring"""
    id: Optional[str] = None
    user_id: str
    content: str
    summary: str = ""
    importance: float = 1.0
    access_count: int = 0
    timestamp: float = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    embeddings: List[float]


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Operation failed")
    error: str = Field(..., description="Error message")
    traceback: Optional[str] = Field(None, description="Error traceback (debug only)")