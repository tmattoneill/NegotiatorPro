"""
User Profile Management

Handles user profile data, including personal information and API key management.
Provides secure storage and retrieval of user-specific configuration.
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from pydantic import BaseModel, EmailStr, Field, validator
from cryptography.fernet import Fernet
import os
import base64
from dotenv import load_dotenv

from .database import db

load_dotenv()

logger = logging.getLogger(__name__)

# Reserved username for super admin - this user has elevated privileges
SUPER_ADMIN_USERNAME = "admin"


class UserProfileCreate(BaseModel):
    """Request model for creating a user profile."""
    username: str = Field(..., min_length=3, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    role: str = Field(default="user")
    preferred_provider: Optional[str] = Field(None, max_length=50)
    preferred_model: Optional[str] = Field(None, max_length=100)

    @validator('role')
    def validate_role(cls, v):
        """Ensure role is valid."""
        valid_roles = ['admin', 'user', 'viewer']
        if v not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(valid_roles)}")
        return v


class UserProfileUpdate(BaseModel):
    """Request model for updating a user profile."""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    preferred_provider: Optional[str] = Field(None, max_length=50)
    preferred_model: Optional[str] = Field(None, max_length=100)


class UserProfile(BaseModel):
    """User profile response model (excludes sensitive data)."""
    id: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool
    profile_updated_at: Optional[datetime] = None
    has_openai_key: bool = False
    has_anthropic_key: bool = False
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None

    class Config:
        from_attributes = True

    @property
    def is_super_admin(self) -> bool:
        """
        Check if this user is the special super admin.

        The super admin is identified by the reserved username 'admin'.
        This user has elevated privileges beyond regular admin-role users.
        """
        return self.username == SUPER_ADMIN_USERNAME

    def has_admin_privileges(self) -> bool:
        """
        Check if user has admin-level privileges.

        Returns True for both super admin and regular admin-role users.
        """
        return self.role == 'admin' or self.is_super_admin

    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Override dict() to include computed properties.
        """
        data = super().dict(**kwargs)
        data['is_super_admin'] = self.is_super_admin
        return data

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Override model_dump() to include computed properties (Pydantic v2).
        """
        data = super().model_dump(**kwargs)
        data['is_super_admin'] = self.is_super_admin
        return data


class EncryptionManager:
    """
    Manages encryption/decryption of sensitive data like API keys.

    Uses Fernet (symmetric encryption) with a key derived from environment.
    In production, use a proper key management service (AWS KMS, HashiCorp Vault, etc.)
    """

    def __init__(self):
        """Initialize encryption manager with key from environment."""
        # Get or generate encryption key
        encryption_key = os.getenv("ENCRYPTION_KEY")

        if not encryption_key:
            # Generate a new key if not exists (for development)
            logger.warning("No ENCRYPTION_KEY found in environment. Generating new key.")
            logger.warning("Add this to your .env file: ENCRYPTION_KEY=<key>")
            encryption_key = Fernet.generate_key().decode()
            logger.warning(f"Generated key: {encryption_key}")

        # Ensure key is bytes
        if isinstance(encryption_key, str):
            encryption_key = encryption_key.encode()

        self.fernet = Fernet(encryption_key)

    def encrypt(self, data: str) -> str:
        """
        Encrypt a string.

        Args:
            data: Plain text string to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        if not data:
            return None

        encrypted = self.fernet.encrypt(data.encode())
        return encrypted.decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt a string.

        Args:
            encrypted_data: Base64-encoded encrypted string

        Returns:
            Decrypted plain text string
        """
        if not encrypted_data:
            return None

        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None


# Global encryption manager
encryption_manager = EncryptionManager()


class UserProfileManager:
    """
    Manages user profile database operations.

    Provides CRUD operations for user profiles with secure API key storage.
    """

    @staticmethod
    async def create_user(profile_data: UserProfileCreate) -> UserProfile:
        """
        Create a new user profile.

        Args:
            profile_data: User profile creation data

        Returns:
            Created user profile

        Raises:
            ValueError: If username or email already exists, or if trying to
                       create a user with the reserved super admin username
        """
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Prevent creating a user with the reserved super admin username
        # (unless this is the initial super admin creation)
        if profile_data.username == SUPER_ADMIN_USERNAME:
            existing_super_admin = await db.fetchrow(
                "SELECT id FROM users WHERE username = $1",
                SUPER_ADMIN_USERNAME
            )
            if existing_super_admin:
                raise ValueError(f"Username '{SUPER_ADMIN_USERNAME}' is reserved for the super admin")

        # Check if username or email already exists
        existing = await db.fetchrow(
            "SELECT id FROM users WHERE username = $1 OR email = $2",
            profile_data.username,
            profile_data.email
        )

        if existing:
            raise ValueError("Username or email already exists")

        # Hash password
        password_hash = pwd_context.hash(profile_data.password)

        # Encrypt API keys if provided
        openai_key_encrypted = None
        if profile_data.openai_api_key:
            openai_key_encrypted = encryption_manager.encrypt(profile_data.openai_api_key)

        anthropic_key_encrypted = None
        if profile_data.anthropic_api_key:
            anthropic_key_encrypted = encryption_manager.encrypt(profile_data.anthropic_api_key)

        # Insert user
        query = """
            INSERT INTO users (
                id, username, email, password_hash, first_name, last_name,
                openai_api_key, anthropic_api_key, role, profile_updated_at,
                preferred_provider, preferred_model
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP, $10, $11)
            RETURNING id, username, email, first_name, last_name, role,
                      created_at, last_login, is_active, profile_updated_at,
                      (openai_api_key IS NOT NULL) as has_openai_key,
                      (anthropic_api_key IS NOT NULL) as has_anthropic_key,
                      preferred_provider, preferred_model
        """

        user_id = str(uuid.uuid4())

        row = await db.fetchrow(
            query,
            user_id,
            profile_data.username,
            profile_data.email,
            password_hash,
            profile_data.first_name,
            profile_data.last_name,
            openai_key_encrypted,
            anthropic_key_encrypted,
            profile_data.role,
            profile_data.preferred_provider,
            profile_data.preferred_model
        )

        logger.info(f"Created user profile: {profile_data.username}")

        # Convert UUID to string for Pydantic
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['id'])
        return UserProfile(**row_dict)

    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by ID.

        Args:
            user_id: User UUID

        Returns:
            User profile or None if not found
        """
        query = """
            SELECT id, username, email, first_name, last_name, role,
                   created_at, last_login, is_active, profile_updated_at,
                   (openai_api_key IS NOT NULL) as has_openai_key,
                   (anthropic_api_key IS NOT NULL) as has_anthropic_key,
                   preferred_provider, preferred_model
            FROM users
            WHERE id = $1
        """

        row = await db.fetchrow(query, user_id)

        if not row:
            return None

        # Convert UUID to string for Pydantic
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['id'])
        return UserProfile(**row_dict)

    @staticmethod
    async def get_user_by_username(username: str) -> Optional[UserProfile]:
        """
        Get user profile by username.

        Args:
            username: Username

        Returns:
            User profile or None if not found
        """
        query = """
            SELECT id, username, email, first_name, last_name, role,
                   created_at, last_login, is_active, profile_updated_at,
                   (openai_api_key IS NOT NULL) as has_openai_key,
                   (anthropic_api_key IS NOT NULL) as has_anthropic_key,
                   preferred_provider, preferred_model
            FROM users
            WHERE username = $1
        """

        row = await db.fetchrow(query, username)

        if not row:
            return None

        # Convert UUID to string for Pydantic
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['id'])
        return UserProfile(**row_dict)

    @staticmethod
    async def get_user_by_email(email: str) -> Optional[UserProfile]:
        """
        Get user profile by email.

        Args:
            email: Email address

        Returns:
            User profile or None if not found
        """
        query = """
            SELECT id, username, email, first_name, last_name, role,
                   created_at, last_login, is_active, profile_updated_at,
                   (openai_api_key IS NOT NULL) as has_openai_key,
                   (anthropic_api_key IS NOT NULL) as has_anthropic_key,
                   preferred_provider, preferred_model
            FROM users
            WHERE email = $1
        """

        row = await db.fetchrow(query, email)

        if not row:
            return None

        # Convert UUID to string for Pydantic
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['id'])
        return UserProfile(**row_dict)

    @staticmethod
    async def update_user(user_id: str, update_data: UserProfileUpdate) -> Optional[UserProfile]:
        """
        Update user profile.

        Args:
            user_id: User UUID
            update_data: Fields to update

        Returns:
            Updated user profile or None if user not found
        """
        # Build dynamic update query
        updates = []
        params = []
        param_index = 1

        if update_data.first_name is not None:
            updates.append(f"first_name = ${param_index}")
            params.append(update_data.first_name)
            param_index += 1

        if update_data.last_name is not None:
            updates.append(f"last_name = ${param_index}")
            params.append(update_data.last_name)
            param_index += 1

        if update_data.email is not None:
            updates.append(f"email = ${param_index}")
            params.append(update_data.email)
            param_index += 1

        if update_data.openai_api_key is not None:
            encrypted_key = encryption_manager.encrypt(update_data.openai_api_key)
            updates.append(f"openai_api_key = ${param_index}")
            params.append(encrypted_key)
            param_index += 1

        if update_data.anthropic_api_key is not None:
            encrypted_key = encryption_manager.encrypt(update_data.anthropic_api_key)
            updates.append(f"anthropic_api_key = ${param_index}")
            params.append(encrypted_key)
            param_index += 1

        if update_data.preferred_provider is not None:
            updates.append(f"preferred_provider = ${param_index}")
            params.append(update_data.preferred_provider)
            param_index += 1

        if update_data.preferred_model is not None:
            updates.append(f"preferred_model = ${param_index}")
            params.append(update_data.preferred_model)
            param_index += 1

        if not updates:
            # No fields to update
            return await UserProfileManager.get_user_by_id(user_id)

        # Add user_id to params
        params.append(user_id)

        query = f"""
            UPDATE users
            SET {', '.join(updates)}
            WHERE id = ${param_index}
            RETURNING id, username, email, first_name, last_name, role,
                      created_at, last_login, is_active, profile_updated_at,
                      (openai_api_key IS NOT NULL) as has_openai_key,
                      (anthropic_api_key IS NOT NULL) as has_anthropic_key,
                      preferred_provider, preferred_model
        """

        row = await db.fetchrow(query, *params)

        if not row:
            return None

        logger.info(f"Updated user profile: {user_id}")

        # Convert UUID to string for Pydantic
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['id'])
        return UserProfile(**row_dict)

    @staticmethod
    async def get_user_api_keys(user_id: str) -> Dict[str, Optional[str]]:
        """
        Get decrypted API keys for a user.

        Args:
            user_id: User UUID

        Returns:
            Dictionary with 'openai_api_key' and 'anthropic_api_key'
        """
        query = """
            SELECT openai_api_key, anthropic_api_key
            FROM users
            WHERE id = $1
        """

        row = await db.fetchrow(query, user_id)

        if not row:
            return {"openai_api_key": None, "anthropic_api_key": None}

        # Decrypt keys
        openai_key = None
        if row['openai_api_key']:
            openai_key = encryption_manager.decrypt(row['openai_api_key'])

        anthropic_key = None
        if row['anthropic_api_key']:
            anthropic_key = encryption_manager.decrypt(row['anthropic_api_key'])

        return {
            "openai_api_key": openai_key,
            "anthropic_api_key": anthropic_key
        }

    @staticmethod
    async def delete_user(user_id: str) -> bool:
        """
        Delete a user profile.

        Args:
            user_id: User UUID

        Returns:
            True if deleted, False if user not found

        Raises:
            ValueError: If attempting to delete the super admin user
        """
        # Check if this is the super admin user
        user = await UserProfileManager.get_user_by_id(user_id)
        if user and user.username == SUPER_ADMIN_USERNAME:
            logger.warning(f"Attempted to delete protected super admin user: {user_id}")
            raise ValueError("Cannot delete the super admin user")

        result = await db.execute(
            "DELETE FROM users WHERE id = $1",
            user_id
        )

        # Check if any rows were deleted
        deleted = result.endswith("1")

        if deleted:
            logger.info(f"Deleted user profile: {user_id}")

        return deleted

    @staticmethod
    async def get_super_admin() -> Optional[UserProfile]:
        """
        Get the super admin user profile.

        Returns:
            Super admin user profile or None if not found
        """
        return await UserProfileManager.get_user_by_username(SUPER_ADMIN_USERNAME)

    @staticmethod
    async def is_super_admin(user_id: str) -> bool:
        """
        Check if a user ID belongs to the super admin.

        Args:
            user_id: User UUID to check

        Returns:
            True if the user is the super admin
        """
        user = await UserProfileManager.get_user_by_id(user_id)
        return user is not None and user.username == SUPER_ADMIN_USERNAME
