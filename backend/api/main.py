"""
FastAPI Main Application

This is the entry point for the FastAPI backend that replaces Gradio.
It provides REST API endpoints for the NegotiatorPro system.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from dotenv import load_dotenv

from .routes import (
    chat_router, auth_router, health_router, models_router,
    users_router, config_router, negotiations_router,
    personas_router, conversations_router, admin_router
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    """
    from ..database import db

    # Startup
    logger.info("=== Starting NegotiatorPro FastAPI Backend ===")
    logger.info("Initializing database connection...")

    try:
        await db.connect()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.warning("Application will continue but database features will be unavailable")

    logger.info("Initializing RAG system on first request...")

    yield

    # Shutdown
    logger.info("=== Shutting down NegotiatorPro FastAPI Backend ===")
    logger.info("Closing database connection...")
    await db.disconnect()


# Create FastAPI app
app = FastAPI(
    title="NegotiatorPro API",
    description="AI-powered negotiation guidance API using RAG",
    version="1.0.0-poc",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS from environment
# CORS_ALLOWED_ORIGINS should be comma-separated list in production
cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
if cors_origins_env:
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    # Default development origins
    cors_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    logger.info("CORS_ALLOWED_ORIGINS not set, using localhost defaults for development")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with user-friendly messages.

    Provides specific feedback for common validation failures like:
    - String length constraints
    - Missing required fields
    - Type mismatches
    """
    errors = exc.errors()

    # Build user-friendly error messages
    friendly_errors = []
    for error in errors:
        field = error.get("loc", ["unknown"])[-1]  # Get the field name
        error_type = error.get("type", "")

        # Handle string length constraints
        if "string_too_long" in error_type or "too_long" in error_type:
            ctx = error.get("ctx", {})
            max_length = ctx.get("max_length", "unknown")
            actual_length = ctx.get("actual_length", "unknown")
            friendly_errors.append(
                f"Field '{field}': Message length ({actual_length:,} characters) exceeds maximum limit of {max_length:,} characters."
            )

        # Handle string too short
        elif "string_too_short" in error_type or "too_short" in error_type:
            ctx = error.get("ctx", {})
            min_length = ctx.get("min_length", "unknown")
            friendly_errors.append(
                f"Field '{field}': Message must be at least {min_length} characters long."
            )

        # Handle missing required fields
        elif error_type == "missing":
            friendly_errors.append(f"Required field '{field}' is missing.")

        # Handle type errors
        elif "type_error" in error_type:
            expected_type = error.get("ctx", {}).get("expected", "correct type")
            friendly_errors.append(f"Field '{field}': Expected {expected_type}.")

        # Default to original error message
        else:
            friendly_errors.append(f"Field '{field}': {error.get('msg', 'Invalid value')}")

    # Log the full validation error for debugging
    logger.warning(f"Validation error on {request.url.path}: {friendly_errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": friendly_errors[0] if len(friendly_errors) == 1 else friendly_errors,
            "error_type": "validation_error"
        }
    )


# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(users_router)
app.include_router(config_router)
app.include_router(negotiations_router)
app.include_router(personas_router)
app.include_router(conversations_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    """Root endpoint - redirects to API docs"""
    return {
        "message": "NegotiatorPro API",
        "version": "1.0.0-poc",
        "docs": "/api/docs"
    }


if __name__ == "__main__":
    import uvicorn

    # Run with: python -m backend.api.main
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
