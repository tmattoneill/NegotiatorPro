"""
Conversation API Routes

Provides REST endpoints for conversation management within negotiations.
"""
import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from ..models import (
    ConversationCreate, ConversationUpdate,
    ConversationResponse, ConversationDetailResponse,
)
from ... import db_operations as db_ops

logger = logging.getLogger(__name__)

conversations_router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _parse_uuid(value: str, field: str = "id") -> UUID:
    """Parse a path parameter into a UUID, returning HTTP 400 on malformed input
    instead of letting ValueError bubble up as an unhandled 500."""
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {field}")


@conversations_router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(data: ConversationCreate, user_id: str):
    """Create a new conversation within a negotiation."""
    # Verify user owns the negotiation
    negotiation = await db_ops.get_negotiation(data.negotiation_id)
    if not negotiation or str(negotiation['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Negotiation not found")

    try:
        conversation = await db_ops.create_conversation(
            negotiation_id=data.negotiation_id,
            title=data.title
        )
        return ConversationResponse(**conversation)
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@conversations_router.get("/negotiation/{negotiation_id}", response_model=List[ConversationDetailResponse])
async def get_negotiation_conversations(negotiation_id: str, user_id: str):
    """Get all conversations for a negotiation with message counts."""
    # Verify user owns the negotiation
    neg_uuid = _parse_uuid(negotiation_id, "negotiation_id")
    negotiation = await db_ops.get_negotiation(neg_uuid)
    if not negotiation or str(negotiation['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Negotiation not found")

    conversations = await db_ops.get_conversations(neg_uuid)

    # Add message count to each conversation
    detailed_conversations = []
    for conv in conversations:
        conv_detail = dict(conv)
        conv_detail['message_count'] = await db_ops.db.fetchval(
            "SELECT COUNT(*) FROM chat_messages WHERE conversation_id = $1",
            conv['id']
        )
        detailed_conversations.append(ConversationDetailResponse(**conv_detail))

    return detailed_conversations


@conversations_router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: str, user_id: str):
    """Get a specific conversation with message count."""
    conversation = await db_ops.get_conversation_detail(_parse_uuid(conversation_id, "conversation_id"))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Verify ownership via negotiation
    negotiation = await db_ops.get_negotiation(conversation['negotiation_id'])
    if not negotiation or str(negotiation['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(**conversation)


@conversations_router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: str, user_id: str, data: ConversationUpdate):
    """Update a conversation title."""
    conv_uuid = _parse_uuid(conversation_id, "conversation_id")
    conversation = await db_ops.get_conversation(conv_uuid)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Verify ownership via negotiation
    negotiation = await db_ops.get_negotiation(conversation['negotiation_id'])
    if not negotiation or str(negotiation['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if data.title is not None:
        conversation = await db_ops.update_conversation(conv_uuid, data.title)

    return ConversationResponse(**conversation)


@conversations_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, user_id: str):
    """Delete a conversation."""
    conv_uuid = _parse_uuid(conversation_id, "conversation_id")
    conversation = await db_ops.get_conversation(conv_uuid)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Verify ownership via negotiation
    negotiation = await db_ops.get_negotiation(conversation['negotiation_id'])
    if not negotiation or str(negotiation['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db_ops.delete_conversation(conv_uuid)


@conversations_router.get("/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, user_id: str, limit: int = None):
    """Get all messages for a conversation."""
    conv_uuid = _parse_uuid(conversation_id, "conversation_id")
    conversation = await db_ops.get_conversation(conv_uuid)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Verify ownership via negotiation
    negotiation = await db_ops.get_negotiation(conversation['negotiation_id'])
    if not negotiation or str(negotiation['user_id']) != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await db_ops.get_conversation_messages(conv_uuid, limit)
    return messages
