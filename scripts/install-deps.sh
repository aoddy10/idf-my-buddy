#!/bin/bash
# Dependency installation script for IDF My Buddy Travel Assistant

set -e

echo "🔧 Installing dependencies for IDF My Buddy Travel Assistant"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11.0"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${RED}❌ Python 3.11+ is required. Found: $python_version${NC}"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

echo -e "${GREEN}✅ Python version: $python_version${NC}"

# Detect package manager preference
if command -v poetry &> /dev/null; then
    PACKAGE_MANAGER="poetry"
    echo -e "${BLUE}📦 Using Poetry for dependency management${NC}"
elif command -v pip &> /dev/null; then
    PACKAGE_MANAGER="pip"
    echo -e "${BLUE}📦 Using pip for dependency management${NC}"
else
    echo -e "${RED}❌ No package manager found. Please install pip or poetry${NC}"
    exit 1
fi

# Function to install with Poetry
install_with_poetry() {
    echo "📥 Installing dependencies with Poetry..."
    
    # Install Poetry if not found
    if ! command -v poetry &> /dev/null; then
        echo "🔧 Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Configure Poetry
    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project true
    
    # Install dependencies
    echo "📦 Installing production dependencies..."
    poetry install --only=main
    
    if [[ "${1:-}" == "dev" ]]; then
        echo "🛠️  Installing development dependencies..."
        poetry install --with=dev,test,docs
    fi
    
    echo -e "${GREEN}✅ Poetry installation completed${NC}"
}

# Function to install with pip
install_with_pip() {
    echo "📥 Installing dependencies with pip..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        echo "🔧 Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    echo "📦 Installing production dependencies..."
    pip install -r requirements.txt
    
    if [[ "${1:-}" == "dev" ]]; then
        echo "🛠️  Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    echo -e "${GREEN}✅ Pip installation completed${NC}"
}

# Function to install system dependencies
install_system_deps() {
    echo "🔧 Installing system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            echo "📦 Installing apt packages..."
            sudo apt-get update
            sudo apt-get install -y \
                tesseract-ocr \
                tesseract-ocr-eng \
                tesseract-ocr-spa \
                tesseract-ocr-fra \
                tesseract-ocr-deu \
                ffmpeg \
                libmagic1 \
                libpq-dev \
                build-essential
        elif command -v yum &> /dev/null; then
            echo "📦 Installing yum packages..."
            sudo yum install -y \
                tesseract \
                tesseract-langpack-eng \
                tesseract-langpack-spa \
                tesseract-langpack-fra \
                tesseract-langpack-deu \
                ffmpeg \
                file-devel \
                postgresql-devel \
                gcc
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "🍺 Installing Homebrew packages..."
            brew update
            brew install \
                tesseract \
                tesseract-lang \
                ffmpeg \
                libmagic \
                postgresql
        else
            echo -e "${YELLOW}⚠️  Homebrew not found. Please install Tesseract and FFmpeg manually${NC}"
        fi
    fi
    
    echo -e "${GREEN}✅ System dependencies installed${NC}"
}

# Function to download language models
download_models() {
    echo "🤖 Downloading AI models..."
    
    # Activate Python environment
    if [[ "$PACKAGE_MANAGER" == "poetry" ]]; then
        poetry run python -c "
import nltk
import spacy
from transformers import pipeline

# Download NLTK data
print('Downloading NLTK data...')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download spaCy models
print('Downloading spaCy models...')
try:
    spacy.cli.download('en_core_web_sm')
except:
    print('Could not download spaCy model. Install manually with: python -m spacy download en_core_web_sm')

print('Models downloaded successfully!')
"
    else
        source .venv/bin/activate 2>/dev/null || true
        python -c "
import nltk
import spacy

# Download NLTK data
print('Downloading NLTK data...')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download spaCy models  
print('Downloading spaCy models...')
try:
    spacy.cli.download('en_core_web_sm')
except:
    print('Could not download spaCy model. Install manually with: python -m spacy download en_core_web_sm')

print('Models downloaded successfully!')
"
    fi
    
    echo -e "${GREEN}✅ AI models downloaded${NC}"
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    if [[ "$PACKAGE_MANAGER" == "poetry" ]]; then
        poetry run python -c "
import fastapi
import sqlmodel
import openai
import cv2
import whisper
print('✅ All core dependencies are importable')
"
    else
        source .venv/bin/activate 2>/dev/null || true
        python -c "
import fastapi
import sqlmodel
import openai
import cv2
import whisper
print('✅ All core dependencies are importable')
"
    fi
    
    echo -e "${GREEN}✅ Installation verification completed${NC}"
}

# Parse command line arguments
INSTALL_TYPE="prod"
SKIP_SYSTEM=false
SKIP_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_TYPE="dev"
            shift
            ;;
        --skip-system)
            SKIP_SYSTEM=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev           Install development dependencies"
            echo "  --skip-system   Skip system dependency installation"
            echo "  --skip-models   Skip AI model downloads"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Install production dependencies"
            echo "  $0 --dev             # Install with development dependencies"
            echo "  $0 --skip-system     # Skip system packages"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main installation flow
echo "🚀 Starting dependency installation..."
echo "   Type: $INSTALL_TYPE"
echo "   Package Manager: $PACKAGE_MANAGER"
echo ""

# Install system dependencies
if [ "$SKIP_SYSTEM" = false ]; then
    install_system_deps
else
    echo -e "${YELLOW}⚠️  Skipping system dependencies${NC}"
fi

# Install Python dependencies
if [[ "$PACKAGE_MANAGER" == "poetry" ]]; then
    install_with_poetry "$INSTALL_TYPE"
else
    install_with_pip "$INSTALL_TYPE"
fi

# Download models
if [ "$SKIP_MODELS" = false ]; then
    download_models
else
    echo -e "${YELLOW}⚠️  Skipping model downloads${NC}"
fi

# Verify installation
verify_installation

echo ""
echo "🎉 Dependency installation completed!"
echo ""
echo "📋 Next steps:"
if [[ "$PACKAGE_MANAGER" == "poetry" ]]; then
    echo "   • Run development server: poetry run uvicorn app.main:app --reload"
    echo "   • Run tests: poetry run pytest"
    echo "   • Activate shell: poetry shell"
else
    echo "   • Activate environment: source .venv/bin/activate"
    echo "   • Run development server: uvicorn app.main:app --reload"
    echo "   • Run tests: pytest"
fi
echo "   • Configure environment: cp .env.example .env"
echo "   • Start with Docker: ./scripts/dev-start.sh"
echo ""
