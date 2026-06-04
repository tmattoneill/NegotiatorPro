"""
Database CRUD Operations for Personas, Negotiations, and Conversations
"""
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from .database import db


# ============================================================================
# USER PERSONAS
# ============================================================================

async def create_user_persona(
    user_id: UUID,
    name: str,
    role_title: Optional[str] = None,
    organization: Optional[str] = None,
    communication_style: Optional[str] = None,
    negotiation_strengths: Optional[str] = None,
    notes: Optional[str] = None,
    is_default: bool = False,
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None
) -> dict:
    """Create a new user persona."""
    # If setting as default, clear existing default
    if is_default:
        await db.execute(
            "UPDATE user_personas SET is_default = FALSE WHERE user_id = $1",
            user_id
        )

    row = await db.fetchrow(
        """
        INSERT INTO user_personas
        (user_id, name, role_title, organization, communication_style,
         negotiation_strengths, notes, is_default, preferred_provider, preferred_model)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING *
        """,
        user_id, name, role_title, organization, communication_style,
        negotiation_strengths, notes, is_default, preferred_provider, preferred_model
    )
    return dict(row)


async def get_user_personas(user_id: UUID) -> List[dict]:
    """Get all personas for a user."""
    rows = await db.fetch(
        "SELECT * FROM user_personas WHERE user_id = $1 ORDER BY is_default DESC, name",
        user_id
    )
    return [dict(r) for r in rows]


async def get_user_persona(persona_id: UUID) -> Optional[dict]:
    """Get a specific user persona."""
    row = await db.fetchrow("SELECT * FROM user_personas WHERE id = $1", persona_id)
    return dict(row) if row else None


async def update_user_persona(persona_id: UUID, **kwargs) -> Optional[dict]:
    """Update a user persona."""
    if not kwargs:
        return await get_user_persona(persona_id)

    # Handle is_default specially
    if kwargs.get('is_default'):
        persona = await get_user_persona(persona_id)
        if persona:
            await db.execute(
                "UPDATE user_personas SET is_default = FALSE WHERE user_id = $1",
                persona['user_id']
            )

    # Build dynamic update
    fields = []
    values = []
    for i, (key, value) in enumerate(kwargs.items(), start=1):
        fields.append(f"{key} = ${i}")
        values.append(value)
    values.append(persona_id)

    row = await db.fetchrow(
        f"UPDATE user_personas SET {', '.join(fields)} WHERE id = ${len(values)} RETURNING *",
        *values
    )
    return dict(row) if row else None


async def delete_user_persona(persona_id: UUID) -> bool:
    """Delete a user persona."""
    result = await db.execute("DELETE FROM user_personas WHERE id = $1", persona_id)
    return result == "DELETE 1"


# ============================================================================
# PARTNER PERSONAS
# ============================================================================

async def create_partner_persona(
    created_by: UUID,
    name: str,
    role_title: Optional[str] = None,
    company: Optional[str] = None,
    communication_style: Optional[str] = None,
    known_interests: Optional[str] = None,
    batna_estimate: Optional[str] = None,
    relationship_notes: Optional[str] = None,
    is_shared: bool = False,
    negotiation_id: Optional[UUID] = None,
    cloned_from: Optional[UUID] = None
) -> dict:
    """Create a new partner persona.

    Pass negotiation_id to create a private copy scoped to a single negotiation
    (hidden from the partner library); leave it None for a reusable template.
    cloned_from records the template a private copy was seeded from.
    """
    row = await db.fetchrow(
        """
        INSERT INTO partner_personas
        (created_by, name, role_title, company, communication_style,
         known_interests, batna_estimate, relationship_notes, is_shared,
         negotiation_id, cloned_from)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING *
        """,
        created_by, name, role_title, company, communication_style,
        known_interests, batna_estimate, relationship_notes, is_shared,
        negotiation_id, cloned_from
    )
    return dict(row)


async def get_partner_personas(user_id: UUID, include_shared: bool = True) -> List[dict]:
    """Get all partner personas accessible to a user.

    Excludes per-negotiation private copies (negotiation_id IS NOT NULL) so the
    partner library only ever lists reusable templates.
    """
    if include_shared:
        rows = await db.fetch(
            """
            SELECT * FROM partner_personas
            WHERE (created_by = $1 OR is_shared = TRUE) AND negotiation_id IS NULL
            ORDER BY name
            """,
            user_id
        )
    else:
        rows = await db.fetch(
            "SELECT * FROM partner_personas WHERE created_by = $1 AND negotiation_id IS NULL ORDER BY name",
            user_id
        )
    return [dict(r) for r in rows]


async def get_partner_persona(persona_id: UUID) -> Optional[dict]:
    """Get a specific partner persona."""
    row = await db.fetchrow("SELECT * FROM partner_personas WHERE id = $1", persona_id)
    return dict(row) if row else None


async def update_partner_persona(persona_id: UUID, **kwargs) -> Optional[dict]:
    """Update a partner persona."""
    if not kwargs:
        return await get_partner_persona(persona_id)

    fields = []
    values = []
    for i, (key, value) in enumerate(kwargs.items(), start=1):
        fields.append(f"{key} = ${i}")
        values.append(value)
    values.append(persona_id)

    row = await db.fetchrow(
        f"UPDATE partner_personas SET {', '.join(fields)} WHERE id = ${len(values)} RETURNING *",
        *values
    )
    return dict(row) if row else None


async def delete_partner_persona(persona_id: UUID) -> bool:
    """Delete a partner persona."""
    result = await db.execute("DELETE FROM partner_personas WHERE id = $1", persona_id)
    return result == "DELETE 1"


# ============================================================================
# NEGOTIATIONS
# ============================================================================

async def create_negotiation(
    user_id: UUID,
    title: str,
    partner_persona_ids: List[UUID],
    description: Optional[str] = None,
    user_persona_id: Optional[UUID] = None,
    primary_partner_id: Optional[UUID] = None,
    settings: dict = None
) -> dict:
    """Create a new negotiation with at least one partner."""
    import json

    async with db.acquire() as conn:
        # Create the negotiation
        row = await conn.fetchrow(
            """
            INSERT INTO negotiations
            (user_id, title, description, user_persona_id, settings)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            user_id, title, description, user_persona_id,
            json.dumps(settings or {})
        )
        negotiation = dict(row)

        # Add partners
        for i, partner_id in enumerate(partner_persona_ids):
            is_primary = (partner_id == primary_partner_id) or (i == 0 and not primary_partner_id)
            await conn.execute(
                """
                INSERT INTO negotiation_partners (negotiation_id, partner_persona_id, is_primary)
                VALUES ($1, $2, $3)
                """,
                negotiation['id'], partner_id, is_primary
            )

        # Parse settings from JSON string to dict for response validation
        if 'settings' in negotiation and isinstance(negotiation['settings'], str):
            negotiation['settings'] = json.loads(negotiation['settings'])

        # Load partners for response (required by NegotiationResponse model)
        partner_rows = await conn.fetch(
            """
            SELECT pp.* FROM partner_personas pp
            JOIN negotiation_partners np ON np.partner_persona_id = pp.id
            WHERE np.negotiation_id = $1
            ORDER BY np.is_primary DESC, pp.name
            """,
            negotiation['id']
        )
        negotiation['partners'] = [dict(p) for p in partner_rows]

        return negotiation


async def get_negotiations(user_id: UUID, status: Optional[str] = None) -> List[dict]:
    """Get all negotiations for a user, including partner personas."""
    import json

    if status:
        rows = await db.fetch(
            """
            SELECT * FROM negotiations
            WHERE user_id = $1 AND status = $2
            ORDER BY updated_at DESC
            """,
            user_id, status
        )
    else:
        rows = await db.fetch(
            "SELECT * FROM negotiations WHERE user_id = $1 ORDER BY updated_at DESC",
            user_id
        )

    # Convert rows to dicts and parse JSON fields
    negotiations = []
    for r in rows:
        neg = dict(r)
        # Parse settings from JSON string to dict if it's a string
        if 'settings' in neg and isinstance(neg['settings'], str):
            neg['settings'] = json.loads(neg['settings'])

        # CRITICAL FIX: Include partners array for each negotiation
        # This matches what get_negotiation_detail() does (lines 281-290)
        partner_rows = await db.fetch(
            """
            SELECT pp.* FROM partner_personas pp
            JOIN negotiation_partners np ON np.partner_persona_id = pp.id
            WHERE np.negotiation_id = $1
            ORDER BY np.is_primary DESC, pp.name
            """,
            neg['id']
        )
        neg['partners'] = [dict(p) for p in partner_rows]

        negotiations.append(neg)

    return negotiations


async def get_negotiation(negotiation_id: UUID) -> Optional[dict]:
    """Get a specific negotiation."""
    import json

    row = await db.fetchrow("SELECT * FROM negotiations WHERE id = $1", negotiation_id)
    if not row:
        return None

    neg = dict(row)
    # Parse settings from JSON string to dict if it's a string
    if 'settings' in neg and isinstance(neg['settings'], str):
        neg['settings'] = json.loads(neg['settings'])

    return neg


async def get_negotiation_detail(negotiation_id: UUID) -> Optional[dict]:
    """Get negotiation with related entities."""
    negotiation = await get_negotiation(negotiation_id)
    if not negotiation:
        return None

    # Get user persona
    if negotiation.get('user_persona_id'):
        negotiation['user_persona'] = await get_user_persona(negotiation['user_persona_id'])
    else:
        negotiation['user_persona'] = None

    # Get partners
    partner_rows = await db.fetch(
        """
        SELECT pp.* FROM partner_personas pp
        JOIN negotiation_partners np ON np.partner_persona_id = pp.id
        WHERE np.negotiation_id = $1
        ORDER BY np.is_primary DESC, pp.name
        """,
        negotiation_id
    )
    negotiation['partners'] = [dict(r) for r in partner_rows]

    # Get conversation count
    negotiation['conversation_count'] = await db.fetchval(
        "SELECT COUNT(*) FROM conversations WHERE negotiation_id = $1",
        negotiation_id
    )

    return negotiation


async def update_negotiation(negotiation_id: UUID, **kwargs) -> Optional[dict]:
    """Update a negotiation."""
    import json

    if not kwargs:
        return await get_negotiation(negotiation_id)

    # Convert settings dict to JSON string
    if 'settings' in kwargs and isinstance(kwargs['settings'], dict):
        kwargs['settings'] = json.dumps(kwargs['settings'])

    fields = []
    values = []
    for i, (key, value) in enumerate(kwargs.items(), start=1):
        fields.append(f"{key} = ${i}")
        values.append(value)
    values.append(negotiation_id)

    row = await db.fetchrow(
        f"UPDATE negotiations SET {', '.join(fields)} WHERE id = ${len(values)} RETURNING *",
        *values
    )
    return dict(row) if row else None


async def delete_negotiation(negotiation_id: UUID) -> bool:
    """Delete a negotiation."""
    result = await db.execute("DELETE FROM negotiations WHERE id = $1", negotiation_id)
    return result == "DELETE 1"


async def add_negotiation_partner(
    negotiation_id: UUID,
    partner_persona_id: UUID,
    is_primary: bool = False
) -> bool:
    """Add a partner to a negotiation."""
    if is_primary:
        # Clear existing primary
        await db.execute(
            "UPDATE negotiation_partners SET is_primary = FALSE WHERE negotiation_id = $1",
            negotiation_id
        )

    await db.execute(
        """
        INSERT INTO negotiation_partners (negotiation_id, partner_persona_id, is_primary)
        VALUES ($1, $2, $3)
        ON CONFLICT (negotiation_id, partner_persona_id) DO UPDATE SET is_primary = $3
        """,
        negotiation_id, partner_persona_id, is_primary
    )
    return True


async def remove_negotiation_partner(negotiation_id: UUID, partner_persona_id: UUID) -> bool:
    """Remove a partner from a negotiation (must keep at least one)."""
    count = await db.fetchval(
        "SELECT COUNT(*) FROM negotiation_partners WHERE negotiation_id = $1",
        negotiation_id
    )
    if count <= 1:
        return False  # Can't remove last partner

    result = await db.execute(
        "DELETE FROM negotiation_partners WHERE negotiation_id = $1 AND partner_persona_id = $2",
        negotiation_id, partner_persona_id
    )
    return result == "DELETE 1"


# ============================================================================
# CONVERSATIONS
# ============================================================================

async def create_conversation(
    negotiation_id: UUID,
    title: Optional[str] = None
) -> dict:
    """Create a new conversation within a negotiation."""
    row = await db.fetchrow(
        """
        INSERT INTO conversations (negotiation_id, title)
        VALUES ($1, $2)
        RETURNING *
        """,
        negotiation_id, title
    )
    return dict(row)


async def get_conversations(negotiation_id: UUID) -> List[dict]:
    """Get all conversations for a negotiation."""
    rows = await db.fetch(
        "SELECT * FROM conversations WHERE negotiation_id = $1 ORDER BY created_at DESC",
        negotiation_id
    )
    return [dict(r) for r in rows]


async def get_conversation(conversation_id: UUID) -> Optional[dict]:
    """Get a specific conversation."""
    row = await db.fetchrow("SELECT * FROM conversations WHERE id = $1", conversation_id)
    return dict(row) if row else None


async def get_conversation_detail(conversation_id: UUID) -> Optional[dict]:
    """Get conversation with message count."""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        return None

    conversation['message_count'] = await db.fetchval(
        "SELECT COUNT(*) FROM chat_messages WHERE conversation_id = $1",
        conversation_id
    )
    return conversation


async def update_conversation(conversation_id: UUID, title: str) -> Optional[dict]:
    """Update a conversation."""
    row = await db.fetchrow(
        "UPDATE conversations SET title = $1 WHERE id = $2 RETURNING *",
        title, conversation_id
    )
    return dict(row) if row else None


async def delete_conversation(conversation_id: UUID) -> bool:
    """Delete a conversation."""
    result = await db.execute("DELETE FROM conversations WHERE id = $1", conversation_id)
    return result == "DELETE 1"


# ============================================================================
# CHAT MESSAGES
# ============================================================================

async def create_chat_message(
    conversation_id: UUID,
    user_id: UUID,
    role: str,
    content: str,
    session_id: Optional[UUID] = None,
    model: Optional[str] = None,
    tokens_used: Optional[int] = None,
    preprocessing_applied: bool = False
) -> dict:
    """Create a new chat message."""
    row = await db.fetchrow(
        """
        INSERT INTO chat_messages
        (conversation_id, user_id, session_id, role, content, model, tokens_used, preprocessing_applied)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *
        """,
        conversation_id, user_id, session_id, role, content, model, tokens_used, preprocessing_applied
    )
    return dict(row)


async def save_conversation_turn(
    conversation_id: UUID,
    user_id: UUID,
    user_content: str,
    assistant_content: str,
    model: Optional[str] = None,
    preprocessing_applied: bool = False,
) -> None:
    """Persist a user message and its assistant reply atomically.

    Both rows commit together or neither does, so a failed assistant insert
    never leaves an orphaned user message in the conversation.
    """
    insert = """
        INSERT INTO chat_messages
        (conversation_id, user_id, session_id, role, content, model, tokens_used, preprocessing_applied)
        VALUES ($1, $2, NULL, $3, $4, $5, NULL, $6)
    """
    async with db.acquire() as conn:
        async with conn.transaction():
            await conn.execute(insert, conversation_id, user_id, "user", user_content, None, preprocessing_applied)
            await conn.execute(insert, conversation_id, user_id, "assistant", assistant_content, model, preprocessing_applied)


async def get_conversation_messages(conversation_id: UUID, limit: Optional[int] = None) -> List[dict]:
    """Get all messages for a conversation, optionally limited."""
    if limit:
        rows = await db.fetch(
            """
            SELECT * FROM chat_messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2
            """,
            conversation_id, limit
        )
    else:
        rows = await db.fetch(
            "SELECT * FROM chat_messages WHERE conversation_id = $1 ORDER BY created_at ASC",
            conversation_id
        )
    return [dict(r) for r in rows]


async def get_chat_message(message_id: int) -> Optional[dict]:
    """Get a specific chat message."""
    row = await db.fetchrow("SELECT * FROM chat_messages WHERE id = $1", message_id)
    return dict(row) if row else None


async def delete_chat_message(message_id: int) -> bool:
    """Delete a chat message."""
    result = await db.execute("DELETE FROM chat_messages WHERE id = $1", message_id)
    return result == "DELETE 1"


async def delete_conversation_messages(conversation_id: UUID) -> int:
    """Delete all messages in a conversation. Returns count of deleted messages."""
    result = await db.execute("DELETE FROM chat_messages WHERE conversation_id = $1", conversation_id)
    # Extract count from result like "DELETE 5"
    if result.startswith("DELETE "):
        return int(result.split(" ")[1])
    return 0
