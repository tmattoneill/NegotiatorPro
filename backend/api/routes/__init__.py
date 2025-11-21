"""API routes package"""
from .chat import router as chat_router
from .auth import router as auth_router
from .health import router as health_router
from .models import router as models_router
from .users import users_router
from .config import router as config_router
from .negotiations import negotiations_router

__all__ = ["chat_router", "auth_router", "health_router", "models_router", "users_router", "config_router", "negotiations_router"]
