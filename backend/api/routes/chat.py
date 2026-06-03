"""Chat endpoint"""
import asyncio
import logging
import time
from uuid import UUID
from typing import List, Optional, Tuple
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ...rag_engine import EnhancedNegotiationRAG, LLMGenerationError
from ... import db_operations as db_ops
from ...persona_text import is_placeholder
from ..middleware.auth import get_current_user
from ...user_profile import UserProfileManager

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


async def resolve_provider_preferences(
    request_provider: Optional[str],
    request_model: Optional[str],
    conversation_id: Optional[str],
    user_id: Optional[str]
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Resolve provider/model preferences from multiple levels.

    Priority order:
    1. Request-level override (provider/model from API request)
    2. Negotiation-level setting (from negotiation.settings)
    3. User-level default preference (from user profile)
    4. System default (None - let RAG system decide)

    Returns:
        Tuple of (provider, model, source) where source indicates where the preference came from.
    """
    # Level 1: Request-level override takes highest priority
    if request_provider and request_model:
        return request_provider, request_model, "request_override"

    # Level 2: Check negotiation settings if conversation_id provided
    if conversation_id:
        try:
            conv_uuid = UUID(conversation_id)
            conversation = await db_ops.get_conversation(conv_uuid)
            if conversation:
                negotiation = await db_ops.get_negotiation(conversation['negotiation_id'])
                if negotiation and negotiation.get('settings'):
                    settings = negotiation['settings']
                    neg_provider = settings.get('provider')
                    neg_model = settings.get('model')
                    if neg_provider and neg_model:
                        logger.info(f"Using negotiation-level provider preference: {neg_provider}/{neg_model}")
                        return neg_provider, neg_model, "negotiation_settings"
        except (ValueError, Exception) as e:
            logger.warning(f"Could not resolve negotiation settings: {e}")

    # Level 3: Check user profile preferences
    if user_id:
        try:
            user_profile = await UserProfileManager.get_user_by_id(user_id)
            if user_profile:
                user_provider = user_profile.preferred_provider
                user_model = user_profile.preferred_model
                if user_provider and user_model:
                    logger.info(f"Using user-level provider preference: {user_provider}/{user_model}")
                    return user_provider, user_model, "user_default"
        except Exception as e:
            logger.warning(f"Could not resolve user preferences: {e}")

    # Level 4: No preferences set - use system default
    return None, None, "system_default"


def _format_kv(label: str, value: Optional[str]) -> Optional[str]:
    """Return '- **Label**: value' if value is meaningful, else None.

    Placeholder values (N/A, -, none, ...) are treated as empty so they never
    leak into the prompt as "Role: N/A". See backend.persona_text.
    """
    if is_placeholder(value):
        return None
    return f"- **{label}**: {value.strip()}"


async def build_negotiation_briefing(conversation_id: Optional[str]) -> str:
    """
    Load the current negotiation + personas from PostgreSQL and format them as
    a markdown briefing block to prepend to the user's question.

    Returns an empty string when the conversation_id is missing, malformed, or
    points at a negotiation with no useful briefing data.
    """
    if not conversation_id:
        return ""
    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        return ""

    try:
        conversation = await db_ops.get_conversation(conv_uuid)
        if not conversation:
            return ""
        detail = await db_ops.get_negotiation_detail(conversation["negotiation_id"])
        if not detail:
            return ""
    except Exception as exc:
        logger.warning(f"Could not load briefing for conversation {conversation_id}: {exc}")
        return ""

    sections: List[str] = []

    # Negotiation header
    header_lines = [f"**Title**: {detail.get('title', '(untitled)')}"]
    if detail.get("description"):
        header_lines.append(f"**Description**: {detail['description']}")
    sections.append("\n".join(header_lines))

    # User persona
    user_persona = detail.get("user_persona")
    if user_persona:
        bullets = [
            _format_kv("Role", user_persona.get("role_title")),
            _format_kv("Organisation", user_persona.get("organization")),
            _format_kv("Communication style", user_persona.get("communication_style")),
            _format_kv("Strengths", user_persona.get("negotiation_strengths")),
            _format_kv("Notes", user_persona.get("notes")),
        ]
        bullets = [b for b in bullets if b]
        if bullets:
            sections.append(
                f"**You ({user_persona.get('name', 'negotiator')})**\n" + "\n".join(bullets)
            )

    # Partner personas
    partners = detail.get("partners") or []
    for idx, partner in enumerate(partners, start=1):
        bullets = [
            _format_kv("Role", partner.get("role_title")),
            _format_kv("Company", partner.get("company")),
            _format_kv("Communication style", partner.get("communication_style")),
            _format_kv("Known interests", partner.get("known_interests")),
            _format_kv("BATNA estimate", partner.get("batna_estimate")),
            _format_kv("Relationship notes", partner.get("relationship_notes")),
        ]
        bullets = [b for b in bullets if b]
        header = f"**Counterparty {idx}: {partner.get('name', 'partner')}**"
        if bullets:
            sections.append(header + "\n" + "\n".join(bullets))
        else:
            sections.append(header)

    if not sections:
        return ""

    return "## Negotiation Briefing\n\n" + "\n\n".join(sections)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per file


async def process_uploaded_files(files: List[UploadFile]) -> str:
    """
    Process uploaded files and return a formatted context string for the LLM.
    Supports: PDF, DOCX, TXT, CSV. Images are noted but not decoded (no vision model).
    Per-file size is capped at MAX_UPLOAD_BYTES to avoid loading huge files into memory.
    """
    import io
    file_context = []

    for file in files:
        content_type = file.content_type or ""
        filename = (file.filename or "unknown").lower()

        # Bounded read: pull at most one byte past the limit so we can reject
        # oversized uploads without loading the entire file into memory.
        file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file.filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
            )

        try:
            if content_type.startswith("image/"):
                file_context.append(f"[Image attached: {file.filename} — image content not readable by this model]")
                logger.info(f"Image uploaded: {file.filename} ({len(file_bytes)} bytes)")

            elif filename.endswith(".pdf") or content_type == "application/pdf":
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(p.strip() for p in pages if p.strip())
                if text:
                    file_context.append(f"[File: {file.filename}]\n{text}")
                    logger.info(f"PDF extracted: {file.filename} ({len(text)} chars, {len(reader.pages)} pages)")
                else:
                    file_context.append(f"[File: {file.filename} — PDF contains no extractable text]")
                    logger.warning(f"PDF yielded no text: {file.filename}")

            elif filename.endswith((".docx",)) or content_type in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ):
                import docx2txt
                text = docx2txt.process(io.BytesIO(file_bytes)).strip()
                if text:
                    file_context.append(f"[File: {file.filename}]\n{text}")
                    logger.info(f"DOCX extracted: {file.filename} ({len(text)} chars)")
                else:
                    file_context.append(f"[File: {file.filename} — document contains no extractable text]")

            elif filename.endswith((".txt", ".csv", ".md")):
                text = file_bytes.decode("utf-8", errors="replace").strip()
                file_context.append(f"[File: {file.filename}]\n{text}")
                logger.info(f"Text file uploaded: {file.filename} ({len(text)} chars)")

            else:
                logger.warning(f"Unsupported file type, skipping: {file.filename} ({content_type})")

        except Exception as e:
            logger.error(f"Failed to process uploaded file {file.filename}: {e}")
            file_context.append(f"[File: {file.filename} — could not be read: {e}]")

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
    mode: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Process a chat question using the RAG system.
    Supports file uploads (images, txt, csv) and optional database persistence.

    Args:
        question: User's question or negotiation context
        conversation_id: Optional conversation ID to save messages to (requires authentication)
        partner_info: Optional context about negotiation partner
        use_premium_model: Whether to use premium model
        use_preprocessing: Whether to apply text preprocessing
        provider: LLM provider override
        model: Model ID override
        files: Optional uploaded files (images, documents)
        current_user: Optional authenticated user (from JWT token)

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

        # Resolve provider preferences from multiple levels
        user_id = current_user.get('id') if current_user else None
        resolved_provider, resolved_model, preference_source = await resolve_provider_preferences(
            request_provider=provider,
            request_model=model,
            conversation_id=conversation_id,
            user_id=user_id
        )

        if resolved_provider and resolved_model:
            logger.info(f"Model override ({preference_source}): {resolved_provider}/{resolved_model}")
        if files:
            logger.info(f"Files uploaded: {len(files)}")

        # Process uploaded files
        file_context = ""
        if files:
            file_context = await process_uploaded_files(files)

        # Enhance question with the negotiation briefing (loaded server-side
        # from conversation_id), uploaded file context, and the legacy
        # partner_info form field (if the frontend chose to send one).
        enhanced_question = question

        briefing = await build_negotiation_briefing(conversation_id)
        if briefing:
            logger.info(f"Briefing loaded for conversation {conversation_id} ({len(briefing)} chars)")
            enhanced_question = f"{briefing}\n\n---\n\n{enhanced_question}"

        if file_context:
            enhanced_question = f"{file_context}\n\n{enhanced_question}"

        if partner_info and partner_info.strip():
            enhanced_question = f"Additional partner context: {partner_info}\n\n{enhanced_question}"

        # Fetch user's stored API keys to pass through to the LLM
        user_api_keys = None
        if user_id:
            try:
                user_api_keys = await UserProfileManager.get_user_api_keys(user_id)
            except Exception as e:
                logger.warning(f"Could not fetch user API keys: {e}")

        # Process question using existing RAG system. get_advice is synchronous
        # and does a blocking llm.invoke() network call — run it in a worker
        # thread so it doesn't block the event loop and serialize all requests.
        answer, usage = await asyncio.to_thread(
            rag.get_advice,
            question=enhanced_question,
            use_premium_model=use_premium_model,
            use_preprocessing=use_preprocessing,
            override_backend=resolved_provider,
            override_model=resolved_model,
            # NegotiatorPro is the negotiator surface, so default to negotiation
            # mode (scoped RAG + negotiation persona) when the client omits it.
            # Revisit when SalesPro shares this endpoint.
            mode=mode or "negotiation",
            user_api_keys=user_api_keys,
            return_usage=True,
        )

        processing_time = time.time() - start_time

        # Determine which model was used
        if resolved_provider and resolved_model:
            # Provider/model was resolved from preferences
            model_used = f"{resolved_provider}/{resolved_model}"
        elif use_premium_model:
            model_config = rag.backend_manager.get_active_model_config("premium")
            model_used = f"{model_config.get('backend', 'unknown')}/{model_config.get('model', 'unknown')}"
        else:
            model_config = rag.backend_manager.get_active_model_config("default")
            model_used = f"{model_config.get('backend', 'unknown')}/{model_config.get('model', 'unknown')}"

        logger.info(f"Question processed successfully in {processing_time:.2f}s using {model_used}")

        # Save messages to database if a conversation is provided (prefer authenticated user)
        # NOTE: Super admin does not save conversations - chat is for testing only
        is_super_admin = current_user and current_user.get('is_super_admin', False)

        if is_super_admin and conversation_id:
            logger.info("Super admin chat - conversation not persisted (testing mode)")

        if conversation_id and not is_super_admin:
            try:
                conversation_uuid = UUID(conversation_id)

                # Persist only for an authenticated user who actually owns this
                # conversation. We never derive the owner from the conversation
                # itself — doing so let an unauthenticated caller write messages
                # attributed to the conversation's real owner (IDOR / spoofing).
                if not (current_user and current_user.get('id')):
                    logger.info("Unauthenticated chat — messages not persisted")
                else:
                    user_uuid = UUID(current_user['id'])

                    # Verify ownership: conversation → negotiation → user_id
                    conversation = await db_ops.get_conversation(conversation_uuid)
                    negotiation = (
                        await db_ops.get_negotiation(conversation['negotiation_id'])
                        if conversation else None
                    )
                    owns_conversation = bool(
                        negotiation and str(negotiation['user_id']) == str(user_uuid)
                    )

                    if not owns_conversation:
                        logger.warning(
                            "User %s does not own conversation %s — messages not persisted",
                            user_uuid, conversation_id,
                        )
                    else:
                        # Atomic: user + assistant rows commit together or not at all
                        await db_ops.save_conversation_turn(
                            conversation_id=conversation_uuid,
                            user_id=user_uuid,
                            user_content=question,
                            assistant_content=answer,
                            model=model_used,
                            preprocessing_applied=use_preprocessing,
                        )
                        logger.info(
                            "Messages saved to conversation %s (user=%s)",
                            conversation_id, current_user['username'],
                        )
            except ValueError:
                logger.warning(f"Invalid conversation_id format: {conversation_id}")
            except Exception as db_error:
                # Don't fail the request if database save fails
                logger.error(f"Failed to save messages to database: {db_error}", exc_info=True)

        return ChatResponse(
            answer=answer,
            model_used=model_used,
            tokens_used=usage.get("total_tokens") or None,
            processing_time=round(processing_time, 2)
        )

    except HTTPException:
        # Intentional HTTP errors (e.g. 413 oversized upload) must keep their
        # status code, not be masked as a 500 by the generic handler below.
        raise

    except LLMGenerationError as e:
        # LLM/upstream failure (bad key, provider down, etc.). Log details
        # server-side; return a clean 502 the frontend renders as an error state
        # rather than letting the exception text surface as an assistant message.
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model could not generate a response. Check your provider/API key and try again."
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
