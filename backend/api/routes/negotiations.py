"""
Negotiation API Routes

Provides REST endpoints for negotiation management including:
- Negotiation creation and retrieval
- Negotiation updates and deletion
- Partner management
- User authorization checks

NOTE: Super admin (username 'admin') cannot create negotiations - this account
is for system testing only, not for negotiation workflows.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Query, Depends

from ..models.negotiations import (
    NegotiationCreate,
    NegotiationUpdate,
    NegotiationResponse,
    NegotiationDetailResponse,
    NegotiationPartnerAdd,
    NegotiationStatus
)
from ..middleware.auth import get_current_user
from ... import db_operations as db_ops

logger = logging.getLogger(__name__)

# Create router
negotiations_router = APIRouter(
    prefix="/api/negotiations",
    tags=["negotiations"]
)


def check_super_admin_restriction(current_user: Optional[dict], action: str = "perform this action"):
    """
    Check if the current user is the super admin and raise an error if so.

    Super admin cannot create/manage negotiations as this account is for testing only.
    """
    if current_user and current_user.get('is_super_admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Super admin cannot {action}. This account is for system testing only."
        )


@negotiations_router.post("/", response_model=NegotiationResponse, status_code=status.HTTP_201_CREATED)
async def create_negotiation(
    user_id: UUID,
    negotiation_data: NegotiationCreate,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Create a new negotiation for a user.

    Args:
        user_id: User UUID (from authentication)
        negotiation_data: Negotiation creation data (requires at least one partner)
        current_user: Optional authenticated user (from JWT token)

    Returns:
        Created negotiation

    Raises:
        400: If validation fails
        403: If super admin attempts to create negotiation
        500: If database operation fails
    """
    check_super_admin_restriction(current_user, "create negotiations")
    try:
        negotiation = await db_ops.create_negotiation(
            user_id=user_id,
            title=negotiation_data.title,
            partner_persona_ids=negotiation_data.partner_persona_ids,
            description=negotiation_data.description,
            user_persona_id=negotiation_data.user_persona_id,
            primary_partner_id=negotiation_data.primary_partner_id,
            settings=negotiation_data.settings
        )
        logger.info(f"Negotiation created: {negotiation['id']}")
        return negotiation
    except ValueError as e:
        logger.warning(f"Negotiation creation validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create negotiation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create negotiation"
        )


@negotiations_router.get("/{negotiation_id}", response_model=NegotiationDetailResponse)
async def get_negotiation(negotiation_id: UUID, user_id: UUID):
    """
    Get a negotiation by ID with full details (partners, persona, conversation count).

    Args:
        negotiation_id: Negotiation UUID
        user_id: User UUID (from authentication)

    Returns:
        Negotiation with details

    Raises:
        404: If negotiation not found or user not authorized
    """
    try:
        negotiation = await db_ops.get_negotiation_detail(negotiation_id)

        if not negotiation or str(negotiation.get('user_id')) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
            )

        return negotiation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get negotiation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve negotiation"
        )


@negotiations_router.get("/", response_model=List[NegotiationResponse])
async def get_user_negotiations(
    user_id: UUID,
    status_filter: Optional[NegotiationStatus] = Query(None, alias="status")
):
    """
    Get all negotiations for a user.

    Args:
        user_id: User UUID (from authentication)
        status_filter: Optional filter by status (active, paused, closed, won, lost)

    Returns:
        List of negotiations

    Raises:
        500: If database operation fails
    """
    try:
        status_value = status_filter.value if status_filter else None
        negotiations = await db_ops.get_negotiations(
            user_id=user_id,
            status=status_value
        )
        return negotiations
    except Exception as e:
        logger.error(f"Failed to get user negotiations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve negotiations"
        )


@negotiations_router.put("/{negotiation_id}", response_model=NegotiationResponse)
@negotiations_router.patch("/{negotiation_id}", response_model=NegotiationResponse)
async def update_negotiation(
    negotiation_id: UUID,
    user_id: UUID,
    update_data: NegotiationUpdate
):
    """
    Update a negotiation.

    Args:
        negotiation_id: Negotiation UUID
        user_id: User UUID (from authentication)
        update_data: Fields to update

    Returns:
        Updated negotiation

    Raises:
        404: If negotiation not found or user not authorized
        500: If database operation fails
    """
    try:
        # First verify ownership
        existing = await db_ops.get_negotiation(negotiation_id)
        if not existing or str(existing.get('user_id')) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
            )

        # Build update kwargs
        update_kwargs = {}
        if update_data.title is not None:
            update_kwargs['title'] = update_data.title
        if update_data.description is not None:
            update_kwargs['description'] = update_data.description
        if update_data.status is not None:
            update_kwargs['status'] = update_data.status.value
        if update_data.user_persona_id is not None:
            update_kwargs['user_persona_id'] = update_data.user_persona_id
        if update_data.settings is not None:
            update_kwargs['settings'] = update_data.settings

        negotiation = await db_ops.update_negotiation(negotiation_id, **update_kwargs)

        if not negotiation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found"
            )

        logger.info(f"Negotiation updated: {negotiation_id}")
        return negotiation
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Negotiation update validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update negotiation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update negotiation"
        )


@negotiations_router.delete("/{negotiation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_negotiation(negotiation_id: UUID, user_id: UUID):
    """
    Delete a negotiation.

    Args:
        negotiation_id: Negotiation UUID
        user_id: User UUID (from authentication)

    Raises:
        404: If negotiation not found or user not authorized
        500: If database operation fails
    """
    try:
        # First verify ownership
        existing = await db_ops.get_negotiation(negotiation_id)
        if not existing or str(existing.get('user_id')) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
            )

        deleted = await db_ops.delete_negotiation(negotiation_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found"
            )

        logger.info(f"Negotiation deleted: {negotiation_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete negotiation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete negotiation"
        )


# ============================================================================
# PARTNER MANAGEMENT
# ============================================================================

@negotiations_router.post("/{negotiation_id}/partners", status_code=status.HTTP_201_CREATED)
async def add_partner_to_negotiation(
    negotiation_id: UUID,
    user_id: UUID,
    partner_data: NegotiationPartnerAdd
):
    """
    Add a partner to a negotiation.

    Args:
        negotiation_id: Negotiation UUID
        user_id: User UUID (from authentication)
        partner_data: Partner to add

    Raises:
        404: If negotiation not found or user not authorized
        500: If database operation fails
    """
    try:
        # Verify ownership
        existing = await db_ops.get_negotiation(negotiation_id)
        if not existing or str(existing.get('user_id')) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
            )

        await db_ops.add_negotiation_partner(
            negotiation_id=negotiation_id,
            partner_persona_id=partner_data.partner_persona_id,
            is_primary=partner_data.is_primary
        )

        logger.info(f"Partner {partner_data.partner_persona_id} added to negotiation {negotiation_id}")
        return {"status": "success", "message": "Partner added"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add partner: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add partner"
        )


@negotiations_router.delete("/{negotiation_id}/partners/{partner_persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_partner_from_negotiation(
    negotiation_id: UUID,
    partner_persona_id: UUID,
    user_id: UUID
):
    """
    Remove a partner from a negotiation (must keep at least one).

    Args:
        negotiation_id: Negotiation UUID
        partner_persona_id: Partner persona UUID to remove
        user_id: User UUID (from authentication)

    Raises:
        400: If trying to remove the last partner
        404: If negotiation not found or user not authorized
        500: If database operation fails
    """
    try:
        # Verify ownership
        existing = await db_ops.get_negotiation(negotiation_id)
        if not existing or str(existing.get('user_id')) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
            )

        removed = await db_ops.remove_negotiation_partner(negotiation_id, partner_persona_id)

        if not removed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last partner from a negotiation"
            )

        logger.info(f"Partner {partner_persona_id} removed from negotiation {negotiation_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove partner: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove partner"
        )
