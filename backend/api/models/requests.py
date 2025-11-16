"""Request models for FastAPI endpoints"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    partner_info: Optional[str] = Field(None, max_length=5000, description="Optional context about negotiation partner")
    use_premium_model: bool = Field(False, description="Whether to use premium model")
    use_preprocessing: bool = Field(True, description="Whether to apply text preprocessing")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I negotiate a salary increase?",
                "partner_info": "My manager is budget-conscious but values my work.",
                "use_premium_model": False,
                "use_preprocessing": True
            }
        }


class LoginRequest(BaseModel):
    """Request model for admin login"""
    password: str = Field(..., min_length=1, description="Admin password")

    class Config:
        json_schema_extra = {
            "example": {
                "password": "admin123"
            }
        }
