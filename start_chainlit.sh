#!/bin/bash

# Quick Start Script for Chainlit Research Agent
# This script helps you start the Chainlit application easily

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Deep Research Multi-Agent System         ║${NC}"
echo -e "${BLUE}║   Chainlit Server Launcher                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Check if uv is installed
echo -e "${YELLOW}[1/4] Checking uv installation...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed!${NC}"
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the environment to get uv in PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ uv installed${NC}"
else
    echo -e "${GREEN}✓ uv is installed${NC}"
fi

# Check if .env file exists
echo -e "${YELLOW}[2/4] Checking environment variables...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}Warning: .env file not found!${NC}"
    echo -e "${YELLOW}Creating .env template...${NC}"
    cat > .env << EOF
# OpenAI API Key (required)
OPENAI_API_KEY=your-openai-key-here

# Anthropic API Key (optional)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Chainlit Configuration (optional)
CHAINLIT_AUTH_SECRET=your-secret-key-here
EOF
    echo -e "${YELLOW}Please edit .env file and add your API keys!${NC}"
    read -p "Press Enter after adding your keys..."
else
    echo -e "${GREEN}✓ .env file found${NC}"
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-key-here" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY is not set properly!${NC}"
    echo -e "${YELLOW}Please edit .env file and add your OpenAI API key${NC}"
    exit 1
fi

# Generate Chainlit auth secret if not set
if [ -z "$CHAINLIT_AUTH_SECRET" ] || [ "$CHAINLIT_AUTH_SECRET" = "your-secret-key-here" ]; then
    echo -e "${YELLOW}Generating Chainlit authentication secret...${NC}"
    AUTH_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    # Update .env file
    if grep -q "CHAINLIT_AUTH_SECRET=" .env; then
        sed -i "s/CHAINLIT_AUTH_SECRET=.*/CHAINLIT_AUTH_SECRET=$AUTH_SECRET/" .env
    else
        echo "CHAINLIT_AUTH_SECRET=$AUTH_SECRET" >> .env
    fi
    export CHAINLIT_AUTH_SECRET=$AUTH_SECRET
    echo -e "${GREEN}✓ Auth secret generated${NC}"
fi

# Install dependencies if needed
echo -e "${YELLOW}[3/4] Installing dependencies...${NC}"

if [ -f pyproject.toml ]; then
    echo -e "${YELLOW}Syncing project dependencies with uv...${NC}"
    uv sync || {
        echo -e "${RED}Failed to sync dependencies!${NC}"
        echo -e "${YELLOW}Trying to install required packages individually...${NC}"
        uv pip install langgraph langchain langchain-openai langchain-anthropic \
                       langchain_community langchain_tavily pydantic rich \
                       chainlit aiosqlite pandas numpy openai graphrag || {
            echo -e "${RED}Error: Could not install dependencies${NC}"
            exit 1
        }
    }
else
    echo -e "${RED}Error: pyproject.toml not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Determine port
PORT=${1:-8000}

echo -e "${YELLOW}[4/4] Starting Chainlit server...${NC}"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}Server will start on: http://localhost:${PORT}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Features available:${NC}"
echo -e "  🔍 Multi-agent research orchestration"
echo -e "  🛠️  Real-time tool call visualization"
echo -e "  📊 Hierarchical agent interaction display"
echo -e "  💬 Interactive chat interface"
echo -e "  📝 Research note tracking"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start Chainlit with uv
uv run chainlit run chainlit_app.py --port $PORT -w
