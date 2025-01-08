#!/bin/bash

# Step 0: Create a Python virtual environment
echo "Creating Python virtual environment..."
python -m venv .venv || { echo "Failed to create virtual environment"; exit 1; }

# Step 1: Activate the virtual environment
echo "Activating the virtual environment..."
# Use 'source' for Linux/MacOS, and 'Scripts\\activate' for Windows
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Step 2: Change directory to FlagEmbedding_new
echo "Changing directory to FlagEmbedding..."
cd FlagEmbedding || { echo "Directory FlagEmbedding_new not found"; exit 1; }

# Step 3: Install the package in editable mode with 'finetune' extras
echo "Installing the package with extras 'finetune'..."
pip install -e .[finetune] || { echo "Failed to install the package"; exit 1; }

# Step 4: Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 || { echo "Failed to install PyTorch"; exit 1; }

echo "Environment setup completed successfully!"
