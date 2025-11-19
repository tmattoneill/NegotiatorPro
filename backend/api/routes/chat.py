"""Chat endpoint"""
import logging
import time
from fastapi import APIRouter, HTTPException, status

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ...rag_engine import EnhancedNegotiationRAG

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


@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """
    Process a chat question using the RAG system.

    Args:
        request: ChatRequest with question and options

    Returns:
        ChatResponse with AI-generated answer

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()

    try:
        # Get RAG system instance
        rag = get_rag_system()

        logger.info(f"Processing question: {request.question[:50]}...")
        logger.info(f"Premium model: {request.use_premium_model}, Preprocessing: {request.use_preprocessing}")
        if request.provider and request.model:
            logger.info(f"Model override: {request.provider}/{request.model}")

        # Enhance question with partner info if provided
        enhanced_question = request.question
        if request.partner_info and request.partner_info.strip():
            enhanced_question = f"Context about my negotiation partner: {request.partner_info}\n\nMy question: {request.question}"

        # Process question using existing RAG system
        # Pass provider/model overrides if specified
        answer = rag.get_advice(
            question=enhanced_question,
            use_premium_model=request.use_premium_model,
            use_preprocessing=request.use_preprocessing,
            override_backend=request.provider,
            override_model=request.model
        )

        processing_time = time.time() - start_time

        # Determine which model was used
        if request.provider and request.model:
            # User specified explicit override
            model_used = f"{request.provider}/{request.model}"
        elif request.use_premium_model:
            model_config = rag.backend_manager.get_active_model_config("premium")
            model_used = f"{model_config.get('backend', 'unknown')}/{model_config.get('model', 'unknown')}"
        else:
            model_config = rag.backend_manager.get_active_model_config("default")
            model_used = f"{model_config.get('backend', 'unknown')}/{model_config.get('model', 'unknown')}"

        logger.info(f"Question processed successfully in {processing_time:.2f}s using {model_used}")

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
