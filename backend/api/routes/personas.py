"""
Persona API Routes

Provides REST endpoints for user and partner persona management.

NOTE: Super admin (username 'admin') cannot create personas - this account
is for system testing only, not for negotiation workflows.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Query, Depends

from ..models import (
    UserPersonaCreate, UserPersonaUpdate, UserPersonaResponse,
    PartnerPersonaCreate, PartnerPersonaUpdate, PartnerPersonaResponse,
)
from ..middleware.auth import get_current_user
from ... import db_operations as db_ops
from ...user_profile import SUPER_ADMIN_USERNAME
from ...persona_text import meaningful_len, MIN_PERSONA_CHARS

logger = logging.getLogger(__name__)

personas_router = APIRouter(prefix="/api/personas", tags=["personas"])

# Descriptive free-text fields that count toward the per-persona context minimum.
_USER_CONTEXT_FIELDS = (
    "role_title", "organization", "communication_style",
    "negotiation_strengths", "notes",
)
_PARTNER_CONTEXT_FIELDS = (
    "role_title", "company", "communication_style",
    "known_interests", "batna_estimate", "relationship_notes",
)


def _enforce_min_context(merged: dict, fields: tuple, what: str) -> None:
    """Raise 422 if the merged persona has < MIN_PERSONA_CHARS of real context.

    Updates are partial, so the caller merges the incoming fields over the
    existing row before calling this — editing one field can't silently drop a
    persona below the minimum, and a grandfathered thin persona must be brought
    up to the minimum on its next edit.
    """
    total = meaningful_len(*(merged.get(f) for f in fields))
    if total < MIN_PERSONA_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Add at least {MIN_PERSONA_CHARS} characters of context across "
                f"{what} (currently {total})."
            ),
        )


def check_super_admin_restriction(current_user: Optional[dict], action: str = "perform this action"):
    """
    Check if the current user is the super admin and raise an error if so.

    Super admin cannot create/manage personas as this account is for testing only.
    """
    if current_user and current_user.get('is_super_admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Super admin cannot {action}. This account is for system testing only."
        )


# ============================================================================
# USER PERSONAS
# ============================================================================

@personas_router.post("/user", response_model=UserPersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_user_persona(
    user_id: str,
    data: UserPersonaCreate,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Create a new user persona."""
    check_super_admin_restriction(current_user, "create personas")
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
    _enforce_min_context(
        {**existing, **update_data}, _USER_CONTEXT_FIELDS,
        "role, organisation, communication style, strengths and notes",
    )
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
async def create_partner_persona(
    user_id: str,
    data: PartnerPersonaCreate,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Create a new partner persona."""
    check_super_admin_restriction(current_user, "create personas")
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
    _enforce_min_context(
        {**existing, **update_data}, _PARTNER_CONTEXT_FIELDS,
        "role, company, communication style, known interests, BATNA and relationship notes",
    )
    persona = await db_ops.update_partner_persona(UUID(persona_id), **update_data)
    return PartnerPersonaResponse(**persona)


@personas_router.delete("/partner/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_partner_persona(persona_id: str, user_id: str):
    """Delete a partner persona (only owner can delete)."""
    existing = await db_ops.get_partner_persona(UUID(persona_id))
    if not existing or str(existing.get('created_by')) != user_id:
        raise HTTPException(status_code=404, detail="Persona not found or not authorized")

    await db_ops.delete_partner_persona(UUID(persona_id))
