"""Request models for FastAPI endpoints"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., min_length=1, max_length=50000, description="User's question or negotiation context")
    conversation_id: Optional[str] = Field(None, description="Conversation ID to save messages to")
    partner_info: Optional[str] = Field(None, max_length=20000, description="Optional context about negotiation partner")
    use_premium_model: bool = Field(False, description="Whether to use premium model")
    use_preprocessing: bool = Field(True, description="Whether to apply text preprocessing")
    provider: Optional[str] = Field(None, description="LLM provider override (openai, anthropic, ollama, ollama-cloud)")
    model: Optional[str] = Field(None, description="Model ID override (e.g., gpt-4o, claude-3-5-sonnet-20241022)")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I negotiate a salary increase?",
                "partner_info": "My manager is budget-conscious but values my work.",
                "use_premium_model": False,
                "use_preprocessing": True,
                "provider": "openai",
                "model": "gpt-4o-mini"
            }
        }


class LoginRequest(BaseModel):
    """Request model for user login"""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "moneill",
                "password": "mypassword"
            }
        }
