"""
Persona API Routes

Provides REST endpoints for user and partner persona management.
"""
import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Query

from ..models import (
    UserPersonaCreate, UserPersonaUpdate, UserPersonaResponse,
    PartnerPersonaCreate, PartnerPersonaUpdate, PartnerPersonaResponse,
)
from ... import db_operations as db_ops

logger = logging.getLogger(__name__)

personas_router = APIRouter(prefix="/api/personas", tags=["personas"])


# ============================================================================
# USER PERSONAS
# ============================================================================

@personas_router.post("/user", response_model=UserPersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_user_persona(user_id: str, data: UserPersonaCreate):
    """Create a new user persona."""
    try:
        persona = await db_ops.create_user_persona(
            user_id=UUID(user_id),
            name=data.name,
            role_title=data.role_title,
            organization=data.organization,
            communication_style=data.communication_style,
            negotiation_strengths=data.negotiation_strengths,
            notes=data.notes,
            is_default=data.is_default
        )
        return UserPersonaResponse(**persona)
    except Exception as e:
        logger.error(f"Failed to create user persona: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user persona")


@personas_router.get("/user", response_model=List[UserPersonaResponse])
async def get_user_personas(user_id: str):
    """Get all user personas for the authenticated user."""
    personas = await db_ops.get_user_personas(UUID(user_id))
    return [UserPersonaResponse(**p) for p in personas]


@personas_router.get("/user/{persona_id}", response_model=UserPersonaResponse)
async def get_user_persona(persona_id: str, user_id: str):
    """Get a specific user persona."""
    persona = await db_ops.get_user_persona(UUID(persona_id))
    if not persona or str(persona['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Persona not found")
    return UserPersonaResponse(**persona)


@personas_router.patch("/user/{persona_id}", response_model=UserPersonaResponse)
async def update_user_persona(persona_id: str, user_id: str, data: UserPersonaUpdate):
    """Update a user persona."""
    # Verify ownership
    existing = await db_ops.get_user_persona(UUID(persona_id))
    if not existing or str(existing['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Persona not found")

    update_data = data.model_dump(exclude_unset=True)
    persona = await db_ops.update_user_persona(UUID(persona_id), **update_data)
    return UserPersonaResponse(**persona)


@personas_router.delete("/user/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_persona(persona_id: str, user_id: str):
    """Delete a user persona."""
    existing = await db_ops.get_user_persona(UUID(persona_id))
    if not existing or str(existing['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Persona not found")

    await db_ops.delete_user_persona(UUID(persona_id))


# ============================================================================
# PARTNER PERSONAS
# ============================================================================

@personas_router.post("/partner", response_model=PartnerPersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_partner_persona(user_id: str, data: PartnerPersonaCreate):
    """Create a new partner persona."""
    try:
        persona = await db_ops.create_partner_persona(
            created_by=UUID(user_id),
            name=data.name,
            role_title=data.role_title,
            company=data.company,
            communication_style=data.communication_style,
            known_interests=data.known_interests,
            batna_estimate=data.batna_estimate,
            relationship_notes=data.relationship_notes,
            is_shared=data.is_shared
        )
        return PartnerPersonaResponse(**persona)
    except Exception as e:
        logger.error(f"Failed to create partner persona: {e}")
        raise HTTPException(status_code=500, detail="Failed to create partner persona")


@personas_router.get("/partner", response_model=List[PartnerPersonaResponse])
async def get_partner_personas(
    user_id: str,
    include_shared: bool = Query(True, description="Include shared personas from other users")
):
    """Get all partner personas accessible to the user."""
    personas = await db_ops.get_partner_personas(UUID(user_id), include_shared=include_shared)
    return [PartnerPersonaResponse(**p) for p in personas]


@personas_router.get("/partner/{persona_id}", response_model=PartnerPersonaResponse)
async def get_partner_persona(persona_id: str, user_id: str):
    """Get a specific partner persona."""
    persona = await db_ops.get_partner_persona(UUID(persona_id))
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    # Allow access if user created it or it's shared
    if str(persona.get('created_by')) != user_id and not persona.get('is_shared'):
        raise HTTPException(status_code=404, detail="Persona not found")
    return PartnerPersonaResponse(**persona)


@personas_router.patch("/partner/{persona_id}", response_model=PartnerPersonaResponse)
async def update_partner_persona(persona_id: str, user_id: str, data: PartnerPersonaUpdate):
    """Update a partner persona (only owner can update)."""
    existing = await db_ops.get_partner_persona(UUID(persona_id))
    if not existing or str(existing.get('created_by')) != user_id:
        raise HTTPException(status_code=404, detail="Persona not found or not authorized")

    update_data = data.model_dump(exclude_unset=True)
    persona = await db_ops.update_partner_persona(UUID(persona_id), **update_data)
    return PartnerPersonaResponse(**persona)


@personas_router.delete("/partner/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_partner_persona(persona_id: str, user_id: str):
    """Delete a partner persona (only owner can delete)."""
    existing = await db_ops.get_partner_persona(UUID(persona_id))
    if not existing or str(existing.get('created_by')) != user_id:
        raise HTTPException(status_code=404, detail="Persona not found or not authorized")

    await db_ops.delete_partner_persona(UUID(persona_id))
