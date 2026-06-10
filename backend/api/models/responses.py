"""Response models for FastAPI endpoints"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PleaseScore(BaseModel):
    """The PLEASE self-assessment parsed from an ANALYSIS-intent response.

    Each element is graded 1–5; total is their sum (6–30). The model is told to
    revise any draft scoring below 20. `weakest` lists the soft-spot letter codes
    for the gutter's one-line read. Only present on ANALYSIS responses.
    """
    polite: int = Field(..., ge=1, le=5)
    logical: int = Field(..., ge=1, le=5)
    empathetic: int = Field(..., ge=1, le=5)
    assertive: int = Field(..., ge=1, le=5)
    strategic: int = Field(..., ge=1, le=5)
    engaging: int = Field(..., ge=1, le=5)
    total: int = Field(..., ge=6, le=30)
    weakest: List[str] = Field(default_factory=list)


class SourceCitation(BaseModel):
    """A RAG source surfaced in the stats gutter for a given turn."""
    title: str = Field(..., description="Display title, e.g. 'Voss · Never Split the Difference'")
    sub: str = Field(default="", description="Secondary line, e.g. a page reference")
    quote: Optional[str] = Field(None, description="One-line passage from the matched chunk")
    active: bool = Field(default=False, description="True for the top-ranked source")


class LeverageEdge(BaseModel):
    label: str
    confidence: str = Field(default="high", description="'high' or 'low'")


class Leverage(BaseModel):
    mineWeight: float = Field(..., ge=0, le=1)
    theirsWeight: float = Field(..., ge=0, le=1)
    summary: str = ""
    mine: List[LeverageEdge] = Field(default_factory=list)
    theirs: List[LeverageEdge] = Field(default_factory=list)


class PartyCard(BaseModel):
    name: str
    initial: str
    summary: str = ""


class Parties(BaseModel):
    me: PartyCard
    them: PartyCard


class Vitals(BaseModel):
    zopa: Optional[str] = None
    batna: Optional[str] = Field(None, description="'strong' | 'moderate' | 'weak'")
    time: Optional[str] = Field(None, description="'urgent' | 'tight' | 'loose'")


class NegotiationContextResponse(BaseModel):
    """Negotiation-level context for the stats gutter. Parties come from the
    personas; leverage and vitals are inferred from the conversation."""
    leverage: Optional[Leverage] = None
    parties: Optional[Parties] = None
    vitals: Optional[Vitals] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="AI-generated negotiation advice")
    model_used: str = Field(..., description="Name of the LLM model used")
    tokens_used: Optional[int] = Field(None, description="Total tokens used (if available)")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    detected_intent: Optional[str] = Field(None, description="Classified query intent: ANALYSIS | TACTICAL | QUESTION | GENERAL")
    please: Optional[PleaseScore] = Field(None, description="PLEASE self-assessment, present on ANALYSIS responses only")
    sources: List[SourceCitation] = Field(default_factory=list, description="RAG citations for this turn")

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
    user: dict = Field(..., description="Authenticated user profile")


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
