"""API models package"""
from .requests import ChatRequest, LoginRequest
from .responses import ChatResponse, LoginResponse, HealthResponse
from .personas import (
    UserPersonaCreate, UserPersonaUpdate, UserPersonaResponse,
    PartnerPersonaCreate, PartnerPersonaUpdate, PartnerPersonaResponse,
)
from .negotiations import (
    NegotiationStatus,
    NegotiationCreate, NegotiationUpdate, NegotiationResponse, NegotiationDetailResponse,
    NegotiationPartnerAdd,
    ConversationCreate, ConversationUpdate, ConversationResponse, ConversationDetailResponse,
    NegotiationDocumentAdd, NegotiationDocumentResponse,
)

__all__ = [
    "ChatRequest",
    "LoginRequest",
    "ChatResponse",
    "LoginResponse",
    "HealthResponse",
    # Personas
    "UserPersonaCreate",
    "UserPersonaUpdate",
    "UserPersonaResponse",
    "PartnerPersonaCreate",
    "PartnerPersonaUpdate",
    "PartnerPersonaResponse",
    # Negotiations
    "NegotiationStatus",
    "NegotiationCreate",
    "NegotiationUpdate",
    "NegotiationResponse",
    "NegotiationDetailResponse",
    "NegotiationPartnerAdd",
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationResponse",
    "ConversationDetailResponse",
    "NegotiationDocumentAdd",
    "NegotiationDocumentResponse",
]
