"""
Negotiation API Routes

Provides REST endpoints for negotiation management including:
- Negotiation creation and retrieval
- Negotiation updates and deletion
- User authorization checks
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse

from ...negotiation_manager import (
    NegotiationManager,
    Negotiation,
    NegotiationCreate,
    NegotiationUpdate
)

logger = logging.getLogger(__name__)

# Create router
negotiations_router = APIRouter(
    prefix="/api/negotiations",
    tags=["negotiations"]
)


@negotiations_router.post("/", response_model=Negotiation, status_code=status.HTTP_201_CREATED)
async def create_negotiation(user_id: str, negotiation_data: NegotiationCreate):
    """
    Create a new negotiation for a user.

    Args:
        user_id: User UUID (from authentication)
        negotiation_data: Negotiation creation data

    Returns:
        Created negotiation

    Raises:
        400: If validation fails
        500: If database operation fails
    """
    try:
        negotiation = await NegotiationManager.create_negotiation(user_id, negotiation_data)
        logger.info(f"Negotiation created: {negotiation.id}")
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


@negotiations_router.get("/{negotiation_id}", response_model=Negotiation)
async def get_negotiation(negotiation_id: str, user_id: str):
    """
    Get a negotiation by ID.

    Args:
        negotiation_id: Negotiation UUID
        user_id: User UUID (from authentication)

    Returns:
        Negotiation

    Raises:
        404: If negotiation not found or user not authorized
    """
    try:
        negotiation = await NegotiationManager.get_negotiation_by_id(negotiation_id, user_id)

        if not negotiation:
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


@negotiations_router.get("/", response_model=List[Negotiation])
async def get_user_negotiations(
    user_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get all negotiations for a user.

    Args:
        user_id: User UUID (from authentication)
        status_filter: Optional filter by status (active, archived, completed)
        limit: Maximum number of negotiations to return (default: 100)
        offset: Number of negotiations to skip (default: 0)

    Returns:
        List of negotiations

    Raises:
        500: If database operation fails
    """
    try:
        negotiations = await NegotiationManager.get_user_negotiations(
            user_id=user_id,
            status=status_filter,
            limit=limit,
            offset=offset
        )

        return negotiations
    except Exception as e:
        logger.error(f"Failed to get user negotiations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve negotiations"
        )


@negotiations_router.put("/{negotiation_id}", response_model=Negotiation)
@negotiations_router.patch("/{negotiation_id}", response_model=Negotiation)
async def update_negotiation(
    negotiation_id: str,
    user_id: str,
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
        negotiation = await NegotiationManager.update_negotiation(
            negotiation_id=negotiation_id,
            user_id=user_id,
            update_data=update_data
        )

        if not negotiation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
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
async def delete_negotiation(negotiation_id: str, user_id: str):
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
        deleted = await NegotiationManager.delete_negotiation(negotiation_id, user_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Negotiation not found or you don't have access"
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
