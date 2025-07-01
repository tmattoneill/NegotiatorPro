#!/bin/bash

# VectorDB Rebuild Wrapper Script
# Simple wrapper to run the Python rebuild script

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔄 Negotiation Advisor - Vector Database Rebuild${NC}"
echo -e "${YELLOW}This script will rebuild your RAG vector database from scratch${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if rebuild script exists
if [ ! -f "rebuild_vectordb.py" ]; then
    echo -e "${RED}❌ rebuild_vectordb.py not found!${NC}"
    exit 1
fi

# Run the rebuild script
python rebuild_vectordb.py