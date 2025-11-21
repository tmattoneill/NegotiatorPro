"""Persona models for FastAPI endpoints"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# ============================================================================
# USER PERSONAS
# ============================================================================

class UserPersonaBase(BaseModel):
    """Base fields for user persona"""
    name: str = Field(..., min_length=1, max_length=255)
    role_title: Optional[str] = Field(None, max_length=255)
    organization: Optional[str] = Field(None, max_length=255)
    communication_style: Optional[str] = None
    negotiation_strengths: Optional[str] = None
    notes: Optional[str] = None
    is_default: bool = False


class UserPersonaCreate(UserPersonaBase):
    """Request model for creating user persona"""
    pass


class UserPersonaUpdate(BaseModel):
    """Request model for updating user persona"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role_title: Optional[str] = Field(None, max_length=255)
    organization: Optional[str] = Field(None, max_length=255)
    communication_style: Optional[str] = None
    negotiation_strengths: Optional[str] = None
    notes: Optional[str] = None
    is_default: Optional[bool] = None


class UserPersonaResponse(UserPersonaBase):
    """Response model for user persona"""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# PARTNER PERSONAS
# ============================================================================

class PartnerPersonaBase(BaseModel):
    """Base fields for partner persona"""
    name: str = Field(..., min_length=1, max_length=255)
    role_title: Optional[str] = Field(None, max_length=255)
    company: Optional[str] = Field(None, max_length=255)
    communication_style: Optional[str] = None
    known_interests: Optional[str] = None
    batna_estimate: Optional[str] = None
    relationship_notes: Optional[str] = None
    is_shared: bool = False


class PartnerPersonaCreate(PartnerPersonaBase):
    """Request model for creating partner persona"""
    pass


class PartnerPersonaUpdate(BaseModel):
    """Request model for updating partner persona"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role_title: Optional[str] = Field(None, max_length=255)
    company: Optional[str] = Field(None, max_length=255)
    communication_style: Optional[str] = None
    known_interests: Optional[str] = None
    batna_estimate: Optional[str] = None
    relationship_notes: Optional[str] = None
    is_shared: Optional[bool] = None


class PartnerPersonaResponse(PartnerPersonaBase):
    """Response model for partner persona"""
    id: UUID
    created_by: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
