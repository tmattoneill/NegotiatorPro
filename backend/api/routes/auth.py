"""Authentication endpoints"""
import logging
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..models.requests import LoginRequest
from ..models.responses import LoginResponse
from ..middleware.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password
)
from ...admin_config import AdminConfig
from ...user_profile import UserProfileManager, UserProfile, SUPER_ADMIN_USERNAME
from ...database import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Initialize admin config
admin_config = AdminConfig()


class UserLoginRequest(BaseModel):
    """User login request model"""
    username: str
    password: str


class UserLoginResponse(BaseModel):
    """User login response with JWT token and user profile"""
    access_token: str
    token_type: str
    expires_in: int
    user: dict
    is_super_admin: bool = False


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


@router.post("/user-login", response_model=UserLoginResponse)
async def user_login(request: UserLoginRequest):
    """
    User login endpoint with JWT token.

    Args:
        request: UserLoginRequest with username and password

    Returns:
        UserLoginResponse with JWT token and user profile

    Raises:
        HTTPException: If username not found or password is invalid
    """
    try:
        # Get user by username
        user = await UserProfileManager.get_user_by_username(request.username)

        if not user:
            logger.warning(f"Login attempt for non-existent user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Get password hash from database
        password_row = await db.fetchrow(
            "SELECT password_hash FROM users WHERE username = $1",
            request.username
        )

        if not password_row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Verify password
        if not verify_password(request.password, password_row['password_hash']):
            logger.warning(f"Failed login attempt for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Update last_login timestamp
        await db.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = $1",
            request.username
        )

        # Determine if this is the super admin
        is_super_admin = user.username == SUPER_ADMIN_USERNAME

        # Create JWT access token with super admin claim
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user.id,
                "username": user.username,
                "role": user.role,
                "is_super_admin": is_super_admin
            },
            expires_delta=access_token_expires
        )

        logger.info(f"User login successful: {request.username} (super_admin={is_super_admin})")

        # Return token and user profile
        return UserLoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user.dict(),
            is_super_admin=is_super_admin
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
