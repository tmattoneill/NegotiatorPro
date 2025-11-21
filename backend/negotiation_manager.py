"""
Negotiation Manager

Handles CRUD operations for negotiations with PostgreSQL database.
Manages negotiation context, preferences, and relationships.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from .database import db

logger = logging.getLogger(__name__)


class KeyPlayer(BaseModel):
    """Model for a key player in a negotiation"""
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(..., min_length=1, max_length=255)
    notes: Optional[str] = None


class NegotiationCreate(BaseModel):
    """Request model for creating a negotiation"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    visibility: str = Field(default="private")
    preferred_backend: Optional[str] = None
    preferred_model: Optional[str] = None
    background: Optional[str] = None
    key_players: Optional[List[KeyPlayer]] = None

    @validator('visibility')
    def validate_visibility(cls, v):
        """Ensure visibility is valid"""
        valid_visibilities = ['private', 'company']
        if v not in valid_visibilities:
            raise ValueError(f"Visibility must be one of: {', '.join(valid_visibilities)}")
        return v


class NegotiationUpdate(BaseModel):
    """Request model for updating a negotiation"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    visibility: Optional[str] = None
    preferred_backend: Optional[str] = None
    preferred_model: Optional[str] = None
    background: Optional[str] = None
    key_players: Optional[List[KeyPlayer]] = None
    status: Optional[str] = None

    @validator('visibility')
    def validate_visibility(cls, v):
        """Ensure visibility is valid"""
        if v is not None:
            valid_visibilities = ['private', 'company']
            if v not in valid_visibilities:
                raise ValueError(f"Visibility must be one of: {', '.join(valid_visibilities)}")
        return v

    @validator('status')
    def validate_status(cls, v):
        """Ensure status is valid"""
        if v is not None:
            valid_statuses = ['active', 'archived', 'completed']
            if v not in valid_statuses:
                raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v


class Negotiation(BaseModel):
    """Response model for a negotiation"""
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    visibility: str
    preferred_backend: Optional[str] = None
    preferred_model: Optional[str] = None
    background: Optional[str] = None
    key_players: Optional[List[KeyPlayer]] = None
    status: str
    created_at: datetime
    updated_at: datetime

    # Computed fields (to be added later)
    conversation_count: int = 0
    document_count: int = 0
    url_count: int = 0

    class Config:
        from_attributes = True


class NegotiationManager:
    """
    Manages negotiation database operations.

    Provides CRUD operations for negotiations with user isolation.
    """

    @staticmethod
    async def create_negotiation(user_id: str, negotiation_data: NegotiationCreate) -> Negotiation:
        """
        Create a new negotiation for a user.

        Args:
            user_id: User UUID
            negotiation_data: Negotiation creation data

        Returns:
            Created negotiation

        Raises:
            Exception: If database operation fails
        """
        try:
            # Convert key_players to JSONB format
            key_players_json = None
            if negotiation_data.key_players:
                key_players_json = [player.dict() for player in negotiation_data.key_players]

            query = """
                INSERT INTO negotiations (
                    user_id, name, description, visibility,
                    preferred_backend, preferred_model,
                    background, key_players, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'active')
                RETURNING id, user_id, name, description, visibility,
                          preferred_backend, preferred_model,
                          background, key_players, status,
                          created_at, updated_at
            """

            row = await db.fetchrow(
                query,
                user_id,
                negotiation_data.name,
                negotiation_data.description,
                negotiation_data.visibility,
                negotiation_data.preferred_backend,
                negotiation_data.preferred_model,
                negotiation_data.background,
                key_players_json
            )

            # Convert UUID to string for Pydantic
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            row_dict['user_id'] = str(row_dict['user_id'])

            # Parse key_players from JSONB
            if row_dict.get('key_players'):
                row_dict['key_players'] = [KeyPlayer(**player) for player in row_dict['key_players']]

            logger.info(f"Created negotiation {row_dict['id']} for user {user_id}")
            return Negotiation(**row_dict)

        except Exception as e:
            logger.error(f"Failed to create negotiation: {e}")
            raise

    @staticmethod
    async def get_negotiation_by_id(negotiation_id: str, user_id: str) -> Optional[Negotiation]:
        """
        Get a negotiation by ID (with user authorization check).

        Args:
            negotiation_id: Negotiation UUID
            user_id: User UUID (for authorization)

        Returns:
            Negotiation or None if not found/not authorized
        """
        try:
            query = """
                SELECT id, user_id, name, description, visibility,
                       preferred_backend, preferred_model,
                       background, key_players, status,
                       created_at, updated_at
                FROM negotiations
                WHERE id = $1 AND user_id = $2
            """

            row = await db.fetchrow(query, negotiation_id, user_id)

            if not row:
                return None

            # Convert UUID to string for Pydantic
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            row_dict['user_id'] = str(row_dict['user_id'])

            # Parse key_players from JSONB
            if row_dict.get('key_players'):
                row_dict['key_players'] = [KeyPlayer(**player) for player in row_dict['key_players']]

            return Negotiation(**row_dict)

        except Exception as e:
            logger.error(f"Failed to get negotiation: {e}")
            return None

    @staticmethod
    async def get_user_negotiations(
        user_id: str,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Negotiation]:
        """
        Get all negotiations for a user.

        Args:
            user_id: User UUID
            status: Optional filter by status
            limit: Maximum number of negotiations to return
            offset: Number of negotiations to skip

        Returns:
            List of negotiations
        """
        try:
            if status:
                query = """
                    SELECT id, user_id, name, description, visibility,
                           preferred_backend, preferred_model,
                           background, key_players, status,
                           created_at, updated_at
                    FROM negotiations
                    WHERE user_id = $1 AND status = $2
                    ORDER BY updated_at DESC
                    LIMIT $3 OFFSET $4
                """
                rows = await db.fetch(query, user_id, status, limit, offset)
            else:
                query = """
                    SELECT id, user_id, name, description, visibility,
                           preferred_backend, preferred_model,
                           background, key_players, status,
                           created_at, updated_at
                    FROM negotiations
                    WHERE user_id = $1
                    ORDER BY updated_at DESC
                    LIMIT $2 OFFSET $3
                """
                rows = await db.fetch(query, user_id, limit, offset)

            negotiations = []
            for row in rows:
                row_dict = dict(row)
                row_dict['id'] = str(row_dict['id'])
                row_dict['user_id'] = str(row_dict['user_id'])

                # Parse key_players from JSONB
                if row_dict.get('key_players'):
                    row_dict['key_players'] = [KeyPlayer(**player) for player in row_dict['key_players']]

                negotiations.append(Negotiation(**row_dict))

            return negotiations

        except Exception as e:
            logger.error(f"Failed to get user negotiations: {e}")
            return []

    @staticmethod
    async def update_negotiation(
        negotiation_id: str,
        user_id: str,
        update_data: NegotiationUpdate
    ) -> Optional[Negotiation]:
        """
        Update a negotiation (with user authorization check).

        Args:
            negotiation_id: Negotiation UUID
            user_id: User UUID (for authorization)
            update_data: Fields to update

        Returns:
            Updated negotiation or None if not found/not authorized
        """
        try:
            # Build dynamic update query
            updates = []
            params = []
            param_index = 1

            if update_data.name is not None:
                updates.append(f"name = ${param_index}")
                params.append(update_data.name)
                param_index += 1

            if update_data.description is not None:
                updates.append(f"description = ${param_index}")
                params.append(update_data.description)
                param_index += 1

            if update_data.visibility is not None:
                updates.append(f"visibility = ${param_index}")
                params.append(update_data.visibility)
                param_index += 1

            if update_data.preferred_backend is not None:
                updates.append(f"preferred_backend = ${param_index}")
                params.append(update_data.preferred_backend)
                param_index += 1

            if update_data.preferred_model is not None:
                updates.append(f"preferred_model = ${param_index}")
                params.append(update_data.preferred_model)
                param_index += 1

            if update_data.background is not None:
                updates.append(f"background = ${param_index}")
                params.append(update_data.background)
                param_index += 1

            if update_data.key_players is not None:
                key_players_json = [player.dict() for player in update_data.key_players]
                updates.append(f"key_players = ${param_index}")
                params.append(key_players_json)
                param_index += 1

            if update_data.status is not None:
                updates.append(f"status = ${param_index}")
                params.append(update_data.status)
                param_index += 1

            if not updates:
                # No fields to update
                return await NegotiationManager.get_negotiation_by_id(negotiation_id, user_id)

            # Add negotiation_id and user_id to params
            params.append(negotiation_id)
            params.append(user_id)

            query = f"""
                UPDATE negotiations
                SET {', '.join(updates)}
                WHERE id = ${param_index} AND user_id = ${param_index + 1}
                RETURNING id, user_id, name, description, visibility,
                          preferred_backend, preferred_model,
                          background, key_players, status,
                          created_at, updated_at
            """

            row = await db.fetchrow(query, *params)

            if not row:
                return None

            # Convert UUID to string for Pydantic
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            row_dict['user_id'] = str(row_dict['user_id'])

            # Parse key_players from JSONB
            if row_dict.get('key_players'):
                row_dict['key_players'] = [KeyPlayer(**player) for player in row_dict['key_players']]

            logger.info(f"Updated negotiation {negotiation_id}")
            return Negotiation(**row_dict)

        except Exception as e:
            logger.error(f"Failed to update negotiation: {e}")
            return None

    @staticmethod
    async def delete_negotiation(negotiation_id: str, user_id: str) -> bool:
        """
        Delete a negotiation (with user authorization check).

        Args:
            negotiation_id: Negotiation UUID
            user_id: User UUID (for authorization)

        Returns:
            True if deleted, False otherwise
        """
        try:
            query = """
                DELETE FROM negotiations
                WHERE id = $1 AND user_id = $2
            """

            result = await db.execute(query, negotiation_id, user_id)

            # Check if any row was deleted
            deleted = result.split()[-1] == '1'

            if deleted:
                logger.info(f"Deleted negotiation {negotiation_id}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete negotiation: {e}")
            return False
