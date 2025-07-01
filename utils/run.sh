#!/bin/bash

# Negotiation Advisor Startup Script
# This script activates the virtual environment and starts the application

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Negotiation Advisor with Admin Panel${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run:${NC}"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if required files exist
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ Main application file not found!${NC}"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Please create it with your OPENAI_API_KEY${NC}"
fi

# Check if sources directory has documents
if [ -d "sources" ] && [ "$(ls -A sources 2>/dev/null)" ]; then
    echo -e "${GREEN}📚 Found documents in sources directory${NC}"
else
    echo -e "${YELLOW}⚠️  No documents found in sources/ directory${NC}"
fi

# Start the application
echo -e "${GREEN}🌐 Launching Negotiation Advisor...${NC}"
echo -e "${YELLOW}Admin credentials: admin123 (change after first login)${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
echo ""

python main.py
