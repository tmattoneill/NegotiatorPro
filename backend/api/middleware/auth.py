"""JWT authentication middleware"""
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.utils.env_validator import get_required_env

# JWT Configuration - secret is required in production
SECRET_KEY = get_required_env(
    "JWT_SECRET_KEY",
    default="dev-secret-key-DO-NOT-USE-IN-PRODUCTION",
    secret=True
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(get_required_env("JWT_TOKEN_EXPIRE_MINUTES", default="30"))

# HTTP Bearer token scheme (auto_error=False makes it optional)
security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Dictionary of claims to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT token from Authorization header.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Dictionary of token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check if token has expired
        exp = payload.get("exp")
        if exp is None:
            raise credentials_exception

        if datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload

    except JWTError:
        raise credentials_exception


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.

    Args:
        plain_password: Plain text password
        hashed_password: Bcrypt hashed password

    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Bcrypt hashed password
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    """
    Get current authenticated user from JWT token.
    This is OPTIONAL - returns None if no token is provided.

    Args:
        credentials: Optional HTTP Bearer credentials from request

    Returns:
        Dictionary with user information, or None if not authenticated
    """
    # If no credentials provided, return None (user not authenticated)
    if credentials is None:
        return None

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])

        # Check if token has expired
        exp = payload.get("exp")
        if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
            return None

        user_id = payload.get("sub")
        if user_id is None:
            return None

        # Return user information from token
        return {
            "id": user_id,
            "username": payload.get("username"),
            "role": payload.get("role", "user"),
            "is_super_admin": payload.get("is_super_admin", False)
        }
    except JWTError:
        # Invalid token, return None instead of raising exception
        return None


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Require authentication - raises exception if not authenticated.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Dictionary with user information

    Raises:
        HTTPException: If not authenticated
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user(credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def require_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Require admin role - raises exception if not an admin.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Dictionary with user information

    Raises:
        HTTPException: If not authenticated or not an admin
    """
    user = await require_auth(credentials)

    if user.get("role") != "admin" and not user.get("is_super_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    return user


async def require_super_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Require super admin - raises exception if not the super admin.

    The super admin is the special 'admin' user with elevated privileges
    beyond regular admin-role users.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Dictionary with user information

    Raises:
        HTTPException: If not authenticated or not the super admin
    """
    user = await require_auth(credentials)

    if not user.get("is_super_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privileges required"
        )

    return user
