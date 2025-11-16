#!/bin/bash
# Start FastAPI backend server

echo "Starting NegotiatorPro FastAPI Backend..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/api/docs"
echo ""

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
