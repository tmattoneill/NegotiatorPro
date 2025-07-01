#!/bin/bash

# NegotiatorPro Installation Script
# This script sets up NegotiatorPro on your system

set -e  # Exit on any error

echo "🤝 NegotiatorPro Installation Script"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python 3 found: $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        echo "Visit: https://www.python.org/downloads/"
        exit 1
    fi
}

# Check if pip is available
check_pip() {
    print_status "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    elif command -v pip &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf .venv
    fi
    
    python3 -m venv .venv
    print_success "Virtual environment created"
}

# Activate virtual environment and install dependencies
install_dependencies() {
    print_status "Activating virtual environment and installing dependencies..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Create .env file if it doesn't exist
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env file from template"
            print_warning "IMPORTANT: Please edit .env and add your OpenAI API key!"
            echo ""
            echo "To get an OpenAI API key:"
            echo "1. Visit: https://platform.openai.com/api-keys"
            echo "2. Create a new API key"
            echo "3. Edit .env file and replace 'your_api_key_here' with your actual key"
            echo ""
        else
            print_warning ".env.example not found. Creating basic .env file..."
            echo "OPENAI_API_KEY=your_api_key_here" > .env
            echo "GRADIO_SERVER_PORT=7860" >> .env
            print_success "Basic .env file created"
        fi
    else
        print_success ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p sources uploads static .config
    print_success "Directories created"
}

# Check if OpenAI API key is set
check_api_key() {
    print_status "Checking OpenAI API key configuration..."
    
    if [ -f ".env" ]; then
        if grep -q "your_api_key_here" .env; then
            print_warning "OpenAI API key is not configured yet!"
            print_warning "Please edit .env file and add your actual API key before running the application."
        else
            # Check if the key looks valid (starts with sk-)
            API_KEY=$(grep "OPENAI_API_KEY=" .env | cut -d'=' -f2)
            if [[ $API_KEY == sk-* ]]; then
                print_success "OpenAI API key appears to be configured"
            else
                print_warning "OpenAI API key format looks incorrect. Should start with 'sk-'"
            fi
        fi
    fi
}

# Create a simple run script
create_run_script() {
    print_status "Creating run script..."
    
    cat > run_negotiatorpro.sh << 'EOF'
#!/bin/bash

# NegotiatorPro Runner Script
echo "🤝 Starting NegotiatorPro..."

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please run install.sh first."
    exit 1
fi

# Check if API key is configured
if grep -q "your_api_key_here" .env; then
    echo "❌ OpenAI API key not configured. Please edit .env file."
    exit 1
fi

# Run the application
echo "🚀 Launching NegotiatorPro..."
python main.py
EOF
    
    chmod +x run_negotiatorpro.sh
    print_success "Run script created: ./run_negotiatorpro.sh"
}

# Main installation process
main() {
    echo "Starting installation process..."
    echo ""
    
    check_python
    check_pip
    create_venv
    install_dependencies
    setup_env
    create_directories
    check_api_key
    create_run_script
    
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file and add your OpenAI API key"
    echo "2. (Optional) Add PDF negotiation books to the 'sources/' directory"
    echo "3. Run the application:"
    echo "   ./run_negotiatorpro.sh"
    echo "   OR"
    echo "   source .venv/bin/activate && python main.py"
    echo ""
    echo "The application will be available at: http://localhost:7860"
    echo "Default admin password: admin123 (change immediately!)"
    echo ""
    echo "For support, visit: https://github.com/tmattoneill/NegotiatorPro"
}

# Run main function
main