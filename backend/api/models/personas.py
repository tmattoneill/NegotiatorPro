"""Persona models for FastAPI endpoints"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID

from ...persona_text import meaningful_len, MIN_PERSONA_CHARS


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

    @model_validator(mode="after")
    def _require_min_context(self):
        total = meaningful_len(
            self.role_title, self.organization, self.communication_style,
            self.negotiation_strengths, self.notes,
        )
        if total < MIN_PERSONA_CHARS:
            raise ValueError(
                f"Add at least {MIN_PERSONA_CHARS} characters of context across "
                f"role, organisation, communication style, strengths and notes "
                f"(currently {total}) so advice can be tailored to you."
            )
        return self


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

    @model_validator(mode="after")
    def _require_min_context(self):
        total = meaningful_len(
            self.role_title, self.company, self.communication_style,
            self.known_interests, self.batna_estimate, self.relationship_notes,
        )
        if total < MIN_PERSONA_CHARS:
            raise ValueError(
                f"Add at least {MIN_PERSONA_CHARS} characters of context across "
                f"role, company, communication style, known interests, BATNA and "
                f"relationship notes (currently {total}) so strategy can be tailored."
            )
        return self


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
