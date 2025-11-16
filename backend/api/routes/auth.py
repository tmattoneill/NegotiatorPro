"""Authentication endpoints"""
import logging
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status

from ..models.requests import LoginRequest
from ..models.responses import LoginResponse
from ..middleware.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ...admin_config import AdminConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Initialize admin config
admin_config = AdminConfig()


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Admin login endpoint.

    Args:
        request: LoginRequest with password

    Returns:
        LoginResponse with JWT access token

    Raises:
        HTTPException: If password is invalid
    """
    # Verify password using existing AdminConfig
    if not admin_config.verify_password(request.password):
        logger.warning("Failed login attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "admin", "role": "admin"},
        expires_delta=access_token_expires
    )

    logger.info("Admin login successful")

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
