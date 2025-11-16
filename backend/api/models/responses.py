"""Response models for FastAPI endpoints"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="AI-generated negotiation advice")
    model_used: str = Field(..., description="Name of the LLM model used")
    tokens_used: Optional[int] = Field(None, description="Total tokens used (if available)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on expert negotiation principles...",
                "model_used": "gpt-4o-mini",
                "tokens_used": 1234,
                "processing_time": 2.5
            }
        }


class LoginResponse(BaseModel):
    """Response model for login endpoint"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current server time")
    version: str = Field(default="1.0.0-poc", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T12:00:00Z",
                "version": "1.0.0-poc"
            }
        }
