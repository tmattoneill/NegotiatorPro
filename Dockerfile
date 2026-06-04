# Backend-only image. The React app is built separately and served as static
# files by the host nginx (see deploy.sh / deploy/nginx), so there is no frontend
# stage here.

# Stage 1: Build Python backend
FROM python:3.11-slim AS backend-builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from backend builder
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend application code
COPY backend/ /app/backend/
COPY .env* /app/
COPY *.json /app/
COPY run-api.sh /app/

# Create necessary directories with proper permissions
# (sources are mounted via docker-compose volume at runtime)
RUN mkdir -p /app/vectorstore /app/uploads /app/sources && \
    chmod -R 755 /app/vectorstore /app/uploads /app/sources

# Create a non-root user for running the application
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check for FastAPI backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the FastAPI backend
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
