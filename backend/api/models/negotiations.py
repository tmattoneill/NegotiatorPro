"""Negotiation and Conversation models for FastAPI endpoints"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from .personas import PartnerPersonaResponse, UserPersonaResponse


class NegotiationStatus(str, Enum):
    active = "active"
    paused = "paused"
    closed = "closed"
    won = "won"
    lost = "lost"


# ============================================================================
# NEGOTIATIONS
# ============================================================================

class NegotiationBase(BaseModel):
    """Base fields for negotiation"""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    status: NegotiationStatus = NegotiationStatus.active
    user_persona_id: Optional[UUID] = None
    settings: Dict[str, Any] = Field(default_factory=dict)


class NegotiationCreate(BaseModel):
    """Request model for creating negotiation"""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    user_persona_id: Optional[UUID] = None
    partner_persona_ids: List[UUID] = Field(..., min_length=1, description="At least one partner required")
    primary_partner_id: Optional[UUID] = None
    settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('partner_persona_ids')
    @classmethod
    def validate_partners(cls, v):
        if not v:
            raise ValueError('At least one partner persona is required')
        return v


class NegotiationUpdate(BaseModel):
    """Request model for updating negotiation"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[NegotiationStatus] = None
    user_persona_id: Optional[UUID] = None
    settings: Optional[Dict[str, Any]] = None


class NegotiationPartnerAdd(BaseModel):
    """Request to add partner to negotiation"""
    partner_persona_id: UUID
    is_primary: bool = False


class NegotiationResponse(NegotiationBase):
    """Response model for negotiation"""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NegotiationDetailResponse(NegotiationResponse):
    """Detailed response with related entities"""
    user_persona: Optional[UserPersonaResponse] = None
    partners: List[PartnerPersonaResponse] = []
    conversation_count: int = 0


# ============================================================================
# CONVERSATIONS
# ============================================================================

class ConversationBase(BaseModel):
    """Base fields for conversation"""
    title: Optional[str] = Field(None, max_length=255)


class ConversationCreate(ConversationBase):
    """Request model for creating conversation"""
    negotiation_id: UUID


class ConversationUpdate(BaseModel):
    """Request model for updating conversation"""
    title: Optional[str] = Field(None, max_length=255)


class ConversationResponse(ConversationBase):
    """Response model for conversation"""
    id: UUID
    negotiation_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationDetailResponse(ConversationResponse):
    """Detailed response with messages"""
    message_count: int = 0


# ============================================================================
# NEGOTIATION DOCUMENTS
# ============================================================================

class NegotiationDocumentAdd(BaseModel):
    """Request to attach document to negotiation"""
    document_id: UUID
    notes: Optional[str] = None


class NegotiationDocumentResponse(BaseModel):
    """Response for negotiation document"""
    id: UUID
    negotiation_id: UUID
    document_id: UUID
    added_at: datetime
    notes: Optional[str] = None

    class Config:
        from_attributes = True
