#!/bin/bash

# ========================================
# NegotiatorPro First-Time Setup Script
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

prompt_required() {
    local var_name=$1
    local prompt_text=$2
    local default_value=$3
    local is_secret=${4:-false}
    
    while true; do
        if [ -n "$default_value" ]; then
            echo -ne "${CYAN}$prompt_text [${default_value}]: ${NC}"
        else
            echo -ne "${CYAN}$prompt_text: ${NC}"
        fi
        
        if [ "$is_secret" = true ]; then
            read -s value
            echo ""
        else
            read value
        fi
        
        # Use default if empty
        if [ -z "$value" ] && [ -n "$default_value" ]; then
            value="$default_value"
        fi
        
        if [ -n "$value" ]; then
            eval "$var_name='$value'"
            return 0
        else
            print_error "This field is required. Please enter a value."
        fi
    done
}

prompt_optional() {
    local var_name=$1
    local prompt_text=$2
    local default_value=$3
    local is_secret=${4:-false}
    
    if [ -n "$default_value" ]; then
        echo -ne "${CYAN}$prompt_text [${default_value}]: ${NC}"
    else
        echo -ne "${CYAN}$prompt_text (optional, press Enter to skip): ${NC}"
    fi
    
    if [ "$is_secret" = true ]; then
        read -s value
        echo ""
    else
        read value
    fi
    
    # Use default if empty
    if [ -z "$value" ] && [ -n "$default_value" ]; then
        value="$default_value"
    fi
    
    eval "$var_name='$value'"
}

generate_secret() {
    openssl rand -hex 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 64 | head -n 1
}

generate_encryption_key() {
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || \
    openssl rand -base64 32 2>/dev/null || \
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1
}

generate_password() {
    openssl rand -base64 16 2>/dev/null | tr -dc 'a-zA-Z0-9' | head -c 16 || \
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1
}

# ========================================
# Pre-flight Checks
# ========================================
print_header "NegotiatorPro Setup"

echo "This script will configure NegotiatorPro for first-time use."
echo "It will create a .env file with your configuration."
echo ""

# Check for existing .env
if [ -f .env ]; then
    print_warning "An existing .env file was found."
    echo -ne "${YELLOW}Do you want to overwrite it? (y/N): ${NC}"
    read overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled. Your existing .env file was preserved."
        exit 0
    fi
    cp .env .env.backup
    print_info "Backed up existing .env to .env.backup"
fi

# Check prerequisites
print_header "Checking Prerequisites"

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker is installed"
else
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    print_success "Docker Compose is installed"
else
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if user can run Docker
if docker ps &> /dev/null; then
    print_success "Docker daemon is accessible"
else
    print_error "Cannot connect to Docker daemon."
    print_info "Try: sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# ========================================
# Environment Selection
# ========================================
print_header "Environment Configuration"

echo "Select your environment:"
echo "  1) Development (relaxed security, local use)"
echo "  2) Production (strict security, public deployment)"
echo ""
echo -ne "${CYAN}Environment [1]: ${NC}"
read env_choice

if [ "$env_choice" = "2" ]; then
    APP_ENV="production"
    print_info "Configuring for PRODUCTION environment"
else
    APP_ENV="development"
    print_info "Configuring for DEVELOPMENT environment"
fi

# ========================================
# Database Configuration
# ========================================
print_header "Database Configuration"

prompt_optional POSTGRES_DB "PostgreSQL database name" "negotiatorpro"
prompt_optional POSTGRES_USER "PostgreSQL username" "negotiatorpro"

# Generate a secure default password
DEFAULT_DB_PASSWORD=$(generate_password)
echo -e "${CYAN}PostgreSQL password [auto-generated]: ${NC}"
echo -ne "${CYAN}  Press Enter to use: ${DEFAULT_DB_PASSWORD:0:4}...${DEFAULT_DB_PASSWORD: -4} or type your own: ${NC}"
read -s custom_password
echo ""

if [ -n "$custom_password" ]; then
    POSTGRES_PASSWORD="$custom_password"
else
    POSTGRES_PASSWORD="$DEFAULT_DB_PASSWORD"
fi
print_success "Database password set"

# ========================================
# Security Configuration
# ========================================
print_header "Security Configuration"

# JWT Secret
JWT_SECRET_KEY=$(generate_secret)
print_success "JWT secret key generated"

# Encryption Key
ENCRYPTION_KEY=$(generate_encryption_key)
print_success "Encryption key generated"

# Admin Password
if [ "$APP_ENV" = "production" ]; then
    prompt_required ADMIN_PASSWORD "Admin password (min 8 characters)" "" true
else
    DEFAULT_ADMIN_PASSWORD=$(generate_password)
    echo -e "${CYAN}Admin password [auto-generated]: ${NC}"
    echo -ne "${CYAN}  Press Enter to use: ${DEFAULT_ADMIN_PASSWORD} or type your own: ${NC}"
    read -s custom_admin_password
    echo ""
    
    if [ -n "$custom_admin_password" ]; then
        ADMIN_PASSWORD="$custom_admin_password"
    else
        ADMIN_PASSWORD="$DEFAULT_ADMIN_PASSWORD"
    fi
fi
print_success "Admin password set"

prompt_optional ADMIN_EMAIL "Admin email" "admin@negotiatorpro.local"

# ========================================
# CORS Configuration
# ========================================
print_header "CORS Configuration"

if [ "$APP_ENV" = "production" ]; then
    print_info "For production, enter your domain(s) where the app will be hosted."
    prompt_required CORS_ALLOWED_ORIGINS "Allowed origins (comma-separated)" "https://yourdomain.com"
else
    prompt_optional CORS_ALLOWED_ORIGINS "Allowed origins" "http://localhost:5173,http://localhost:3000"
fi

# ========================================
# LLM Backend Configuration
# ========================================
print_header "LLM Backend Configuration"

echo "Configure your AI model providers. You need at least one."
echo ""

# OpenAI
echo -e "${YELLOW}OpenAI (GPT-4, GPT-3.5)${NC}"
prompt_optional OPENAI_API_KEY "OpenAI API Key (starts with sk-)" "" true
if [ -n "$OPENAI_API_KEY" ]; then
    print_success "OpenAI configured"
fi

# Anthropic
echo ""
echo -e "${YELLOW}Anthropic (Claude)${NC}"
prompt_optional ANTHROPIC_API_KEY "Anthropic API Key (starts with sk-ant-)" "" true
if [ -n "$ANTHROPIC_API_KEY" ]; then
    print_success "Anthropic configured"
fi

# Ollama
echo ""
echo -e "${YELLOW}Ollama (Local/Self-hosted models)${NC}"
echo "Ollama allows you to run models locally on your machine."

if command -v ollama &> /dev/null || curl -s http://localhost:11434/api/version &> /dev/null; then
    print_success "Ollama detected on this system"
    OLLAMA_DETECTED=true
else
    print_info "Ollama not detected. You can install it later from https://ollama.com"
    OLLAMA_DETECTED=false
fi

prompt_optional OLLAMA_BASE_URL "Ollama URL" "http://host.docker.internal:11434"

echo ""
echo -e "${YELLOW}Ollama Cloud (optional)${NC}"
prompt_optional OLLAMA_CLOUD_URL "Ollama Cloud URL" "https://ollama.com"
prompt_optional OLLAMA_API_KEY "Ollama Cloud API Key" "" true

# Validate at least one LLM is configured
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ "$OLLAMA_DETECTED" = false ] && [ -z "$OLLAMA_API_KEY" ]; then
    print_warning "No LLM provider configured. The app will have limited functionality."
    echo -ne "${YELLOW}Continue anyway? (y/N): ${NC}"
    read continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled. Please configure at least one LLM provider."
        exit 0
    fi
fi

# ========================================
# Write .env File
# ========================================
print_header "Creating Configuration"

cat > .env << EOF
# ========================================
# NegotiatorPro Environment Configuration
# Generated by setup.sh on $(date)
# ========================================

# ========================================
# Application Environment
# ========================================
APP_ENV=${APP_ENV}

# ========================================
# Security
# ========================================
JWT_SECRET_KEY=${JWT_SECRET_KEY}
ADMIN_PASSWORD=${ADMIN_PASSWORD}
ADMIN_EMAIL=${ADMIN_EMAIL}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# ========================================
# CORS Configuration
# ========================================
CORS_ALLOWED_ORIGINS=${CORS_ALLOWED_ORIGINS}

# ========================================
# Database Configuration (PostgreSQL)
# ========================================
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

# ========================================
# LLM Backend Configuration
# ========================================
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
OLLAMA_CLOUD_URL=${OLLAMA_CLOUD_URL}
OLLAMA_API_KEY=${OLLAMA_API_KEY}

# ========================================
# Docker Port Configuration (optional)
# ========================================
# Override default ports if needed (e.g., to avoid conflicts)
# POSTGRES_HOST_PORT=5432
# API_HOST_PORT=8000
# FRONTEND_HOST_PORT=5173
EOF

print_success ".env file created"

# ========================================
# Create Data Directories
# ========================================
print_header "Creating Data Directories"

# We now use a Docker named volume for Postgres (postgres_data),
# so no need to create ./data/db here.
mkdir -p data/vectorstore data/uploads data/sources data/config
print_success "Data directories created"

# ========================================
# Build and Start
# ========================================
print_header "Starting NegotiatorPro"

echo "Ready to build and start the application."
echo -ne "${CYAN}Start now? (Y/n): ${NC}"
read start_now

if [[ "$start_now" =~ ^[Nn]$ ]]; then
    print_info "Setup complete. Run 'docker compose up -d --build' when ready."
else
    print_info "Building containers (this may take a few minutes)..."
    docker compose up -d --build
    
    echo ""
    print_info "Waiting for services to start..."
    sleep 10
    
    # Check health
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        print_success "Backend is running"
    else
        print_warning "Backend may still be starting. Check 'docker compose logs backend'"
    fi

    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        print_success "Frontend is running"
    else
        print_warning "Frontend may still be starting. Check 'docker compose logs frontend'"
    fi
fi

# ========================================
# Summary
# ========================================
print_header "Setup Complete!"

echo -e "${GREEN}NegotiatorPro has been configured successfully.${NC}"
echo ""
echo "Access the application:"
echo -e "  ${CYAN}Frontend:${NC} http://localhost:5173"
echo -e "  ${CYAN}Backend API:${NC} http://localhost:8000"
echo -e "  ${CYAN}API Docs:${NC} http://localhost:8000/docs"
echo ""
echo "Admin credentials:"
echo -e "  ${CYAN}Email:${NC} ${ADMIN_EMAIL}"
echo -e "  ${CYAN}Password:${NC} (as configured above)"
echo ""
echo "Useful commands:"
echo -e "  ${CYAN}View logs:${NC} docker compose logs -f"
echo -e "  ${CYAN}Stop:${NC} docker compose down"
echo -e "  ${CYAN}Restart:${NC} docker compose restart"
echo -e "  ${CYAN}Rebuild:${NC} docker compose up -d --build"
echo ""

if [ -n "$POSTGRES_PASSWORD" ] && [ "$POSTGRES_PASSWORD" = "$DEFAULT_DB_PASSWORD" ]; then
    print_warning "Database password was auto-generated. It's stored in your .env file."
fi

if [ "$APP_ENV" = "development" ]; then
    print_warning "Running in DEVELOPMENT mode. Do not use in production without reconfiguring."
fi

echo ""
print_success "Happy negotiating!"
