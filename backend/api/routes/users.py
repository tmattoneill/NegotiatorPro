"""
User Profile API Routes

Provides REST endpoints for user profile management including:
- User creation and registration
- Profile viewing and updates
- API key management
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from ...user_profile import (
    UserProfileManager,
    UserProfile,
    UserProfileCreate,
    UserProfileUpdate
)

logger = logging.getLogger(__name__)

# Create router
users_router = APIRouter(
    prefix="/api/users",
    tags=["users"]
)


@users_router.post("/", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def create_user(profile_data: UserProfileCreate):
    """
    Create a new user profile.

    Creates a user with:
    - Basic profile information (name, email, username)
    - Optional API keys (encrypted)
    - Role-based access control

    Returns:
        Created user profile (excluding sensitive data)

    Raises:
        400: If username or email already exists
        500: If database operation fails
    """
    try:
        user = await UserProfileManager.create_user(profile_data)
        logger.info(f"User created: {user.username}")
        return user
    except ValueError as e:
        logger.warning(f"User creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user profile"
        )


@users_router.get("/{user_id}", response_model=UserProfile)
async def get_user(user_id: str):
    """
    Get user profile by ID.

    Args:
        user_id: User UUID

    Returns:
        User profile (excluding sensitive data)

    Raises:
        404: If user not found
    """
    try:
        user = await UserProfileManager.get_user_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@users_router.get("/username/{username}", response_model=UserProfile)
async def get_user_by_username(username: str):
    """
    Get user profile by username.

    Args:
        username: Username

    Returns:
        User profile (excluding sensitive data)

    Raises:
        404: If user not found
    """
    try:
        user = await UserProfileManager.get_user_by_username(username)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user by username: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@users_router.get("/email/{email}", response_model=UserProfile)
async def get_user_by_email(email: str):
    """
    Get user profile by email.

    Args:
        email: Email address

    Returns:
        User profile (excluding sensitive data)

    Raises:
        404: If user not found
    """
    try:
        user = await UserProfileManager.get_user_by_email(email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user by email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@users_router.patch("/{user_id}", response_model=UserProfile)
async def update_user(user_id: str, update_data: UserProfileUpdate):
    """
    Update user profile.

    Allows updating:
    - Name fields
    - Email address
    - API keys

    Args:
        user_id: User UUID
        update_data: Fields to update

    Returns:
        Updated user profile

    Raises:
        404: If user not found
        500: If database operation fails
    """
    try:
        user = await UserProfileManager.update_user(user_id, update_data)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        logger.info(f"User updated: {user_id}")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@users_router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    """
    Delete user profile.

    Args:
        user_id: User UUID

    Raises:
        404: If user not found
        500: If database operation fails
    """
    try:
        deleted = await UserProfileManager.delete_user(user_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        logger.info(f"User deleted: {user_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user profile"
        )


@users_router.get("/{user_id}/api-keys")
async def get_user_api_keys(user_id: str):
    """
    Get user's API keys (decrypted).

    NOTE: This endpoint should be protected with proper authentication
    and only allow users to access their own keys.

    Args:
        user_id: User UUID

    Returns:
        Dictionary with openai_api_key and anthropic_api_key

    Raises:
        404: If user not found
    """
    try:
        # First verify user exists
        user = await UserProfileManager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Get decrypted keys
        keys = await UserProfileManager.get_user_api_keys(user_id)

        return {
            "openai_api_key": keys["openai_api_key"] if keys["openai_api_key"] else None,
            "anthropic_api_key": keys["anthropic_api_key"] if keys["anthropic_api_key"] else None,
            "has_openai_key": keys["openai_api_key"] is not None,
            "has_anthropic_key": keys["anthropic_api_key"] is not None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys"
        )
