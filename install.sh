#!/bin/bash

# DermaHelper Installation Script
# This script sets up the environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Installing DermaHelper..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is too old. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing DermaHelper in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p models
mkdir -p data

# Set up Kaggle credentials (if not already set)
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle credentials not found."
    echo "Please set up your Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Scroll to API section and click 'Create New API Token'"
    echo "3. Download kaggle.json and place it in ~/.kaggle/"
    echo "4. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
fi

echo "✅ Installation completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "python example_logging.py"
echo ""
echo "To start preprocessing data:"
echo "python preprocessing.py" 