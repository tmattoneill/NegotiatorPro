"""
FastAPI Main Application

This is the entry point for the FastAPI backend that replaces Gradio.
It provides REST API endpoints for the NegotiatorPro system.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routes import chat_router, auth_router, health_router

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
    # Startup
    logger.info("=== Starting NegotiatorPro FastAPI Backend ===")
    logger.info("Initializing RAG system on first request...")

    yield

    # Shutdown
    logger.info("=== Shutting down NegotiatorPro FastAPI Backend ===")


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

# Configure CORS
# In production, replace with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)


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
