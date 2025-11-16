"""API routes package"""
from .chat import router as chat_router
from .auth import router as auth_router
from .health import router as health_router

__all__ = ["chat_router", "auth_router", "health_router"]
