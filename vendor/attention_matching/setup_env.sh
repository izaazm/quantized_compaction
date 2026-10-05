#!/bin/bash
set -e

echo "Installing uv..."
pip install uv

echo "Creating virtual environment with Python 3.12..."
uv venv --python 3.12

echo "Activating virtual environment..."
# For the current shell script execution only
source .venv/bin/activate

echo "Installing cartridges library in editable mode..."
cd ../cartridges
uv pip install -e .
cd ../compaction

echo "Installing compaction requirements..."
uv pip install -r requirements.txt

uv pip install hf_transfer

echo ""
echo "=================================================================="
echo "Environment setup complete!"
echo "Please run the following command to activate your new environment:"
echo ""
echo "    source .venv/bin/activate"
echo "=================================================================="
