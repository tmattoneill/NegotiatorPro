"""Authentication endpoints"""
import logging
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status

from ..models.requests import LoginRequest
from ..models.responses import LoginResponse
from ..middleware.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password
)
from ...user_profile import UserProfileManager
from ...database import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Unified login endpoint. Admin is just a user with role='admin'.
    """
    try:
        user = await UserProfileManager.get_user_by_username(request.username)
        if not user:
            logger.warning(f"Login attempt for unknown user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        password_row = await db.fetchrow(
            "SELECT password_hash FROM users WHERE username = $1",
            request.username
        )
        if not password_row or not verify_password(request.password, password_row['password_hash']):
            logger.warning(f"Failed login for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        await db.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = $1",
            request.username
        )

        access_token = create_access_token(
            data={"sub": user.id, "username": user.username, "role": user.role},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        logger.info(f"Login successful: {request.username} (role={user.role})")

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user.dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
