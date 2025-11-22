"""Chat endpoint"""
import logging
import time
import base64
from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ...rag_engine import EnhancedNegotiationRAG
from ... import db_operations as db_ops
from ..middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# Initialize RAG system (singleton)
rag_system = None


def get_rag_system():
    """Get or initialize RAG system"""
    global rag_system
    if rag_system is None:
        logger.info("Initializing RAG system...")
        rag_system = EnhancedNegotiationRAG()
        rag_system.setup_system()
        logger.info("RAG system initialized successfully")
    return rag_system


async def process_uploaded_files(files: List[UploadFile]) -> str:
    """
    Process uploaded files and return formatted context string.

    Supports:
    - Images: Convert to base64 for vision models
    - Text files (.txt, .csv): Extract text content
    """
    file_context = []

    for file in files:
        content_type = file.content_type or ""
        filename = file.filename or "unknown"

        # Read file content
        file_bytes = await file.read()

        if content_type.startswith("image/"):
            # For images, encode as base64
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            file_context.append(f"[Image: {filename}]")
            # Note: We're adding this as context. For vision models, we'd need to format differently
            # For now, we just note that an image was uploaded
            logger.info(f"Image uploaded: {filename} ({len(file_bytes)} bytes)")

        elif filename.endswith(('.txt', '.csv')):
            # Extract text from text files
            try:
                text_content = file_bytes.decode('utf-8')
                file_context.append(f"[File: {filename}]\n{text_content}")
                logger.info(f"Text file uploaded: {filename} ({len(text_content)} chars)")
            except UnicodeDecodeError:
                logger.warning(f"Could not decode file: {filename}")
                file_context.append(f"[File: {filename} - could not read content]")

    return "\n\n".join(file_context) if file_context else ""


@router.post("/chat", response_model=ChatResponse)
async def process_chat(
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    partner_info: Optional[str] = Form(None),
    use_premium_model: bool = Form(False),
    use_preprocessing: bool = Form(True),
    provider: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Process a chat question using the RAG system.
    Supports file uploads (images, txt, csv) and saves to database.

    Args:
        question: User's question or negotiation context
        conversation_id: Optional conversation ID to save messages to
        partner_info: Optional context about negotiation partner
        use_premium_model: Whether to use premium model
        use_preprocessing: Whether to apply text preprocessing
        provider: LLM provider override
        model: Model ID override
        files: Optional uploaded files (images, documents)
        current_user: Authenticated user (injected by dependency)

    Returns:
        ChatResponse with AI-generated answer

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()

    try:
        # Get RAG system instance
        rag = get_rag_system()

        logger.info(f"Processing question: {question[:50]}...")
        logger.info(f"Premium model: {use_premium_model}, Preprocessing: {use_preprocessing}")
        if provider and model:
            logger.info(f"Model override: {provider}/{model}")
        if files:
            logger.info(f"Files uploaded: {len(files)}")

        # Process uploaded files
        file_context = ""
        if files:
            file_context = await process_uploaded_files(files)

        # Enhance question with partner info and file context if provided
        enhanced_question = question

        if file_context:
            enhanced_question = f"{file_context}\n\n{enhanced_question}"

        if partner_info and partner_info.strip():
            enhanced_question = f"Context about my negotiation partner: {partner_info}\n\n{enhanced_question}"

        # Process question using existing RAG system
        # Pass provider/model overrides if specified
        answer = rag.get_advice(
            question=enhanced_question,
            use_premium_model=use_premium_model,
            use_preprocessing=use_preprocessing,
            override_backend=provider,
            override_model=model
        )

        processing_time = time.time() - start_time

        # Determine which model was used
        if provider and model:
            # User specified explicit override
            model_used = f"{provider}/{model}"
        elif use_premium_model:
            model_config = rag.backend_manager.get_active_model_config("premium")
            model_used = f"{model_config.get('backend', 'unknown')}/{model_config.get('model', 'unknown')}"
        else:
            model_config = rag.backend_manager.get_active_model_config("default")
            model_used = f"{model_config.get('backend', 'unknown')}/{model_config.get('model', 'unknown')}"

        logger.info(f"Question processed successfully in {processing_time:.2f}s using {model_used}")

        # Save messages to database if conversation_id is provided
        conversation_uuid: Optional[UUID] = None
        if conversation_id:
            try:
                conversation_uuid = UUID(conversation_id)
                user_uuid = UUID(current_user['id'])

                # Save user message
                await db_ops.create_chat_message(
                    conversation_id=conversation_uuid,
                    user_id=user_uuid,
                    role="user",
                    content=question,
                    preprocessing_applied=use_preprocessing
                )

                # Save assistant response
                await db_ops.create_chat_message(
                    conversation_id=conversation_uuid,
                    user_id=user_uuid,
                    role="assistant",
                    content=answer,
                    model=model_used,
                    preprocessing_applied=use_preprocessing
                )

                logger.info(f"Messages saved to conversation {conversation_id}")
            except ValueError as ve:
                logger.warning(f"Invalid conversation_id format: {conversation_id}")
            except Exception as db_error:
                # Don't fail the request if database save fails
                logger.error(f"Failed to save messages to database: {db_error}", exc_info=True)

        return ChatResponse(
            answer=answer,
            model_used=model_used,
            tokens_used=None,  # TODO: Extract from LLM response metadata
            processing_time=round(processing_time, 2)
        )

    except Exception as e:
        # Log full error details server-side for debugging
        logger.error(f"Error processing question: {str(e)}", exc_info=True)

        # Return generic error message to client (security best practice)
        # Don't expose internal implementation details or stack traces
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question. Please try again or contact support if the issue persists."
        )
