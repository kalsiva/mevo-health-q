#!/bin/bash
# Setup script for Chest X-ray Tokenizer

set -e  # Exit on error

echo "=========================================="
echo "Chest X-ray Tokenizer - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi
echo "✓ pip3 is available"

# Create virtual environment (optional but recommended)
echo ""
echo "[2/5] Setting up virtual environment (optional)..."
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source venv/bin/activate  (Linux/Mac)"
    echo "  venv\\Scripts\\activate   (Windows)"
    echo ""
    read -p "Press Enter to continue..."
fi

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
echo "This may take a few minutes..."
pip3 install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "[4/5] Creating directories..."
mkdir -p data embeddings models .cache sample_images
echo "✓ Directories created"

# Run quickstart test
echo ""
echo "[5/5] Running quickstart test..."
read -p "Do you want to run the quickstart test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 quickstart_test.py
fi

echo ""
echo "=========================================="
echo "Setup Complete! ✓"
echo "=========================================="
echo ""
echo "Quick commands to try:"
echo "  • View statistics:    python cli.py stats"
echo "  • Add an image:       python cli.py add image.png --pathologies 'Pneumonia'"
echo "  • Infer pathologies:  python cli.py infer image.png"
echo "  • Start API:          python api.py"
echo ""
echo "Read QUICKSTART.md for detailed instructions."
echo ""
