#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# GEAK-OptimAgentV2 Setup Script
# ================================
# This script sets up GEAK-agent and GEAK-eval dependencies.
# The same setup is done automatically by launch_agent.py, but this
# script allows manual setup if needed.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Setting up GEAK-OptimAgentV2 dependencies..."
echo "=============================================="

# 1. Clone GEAK-agent (main branch)
if [ ! -d "GEAK-agent" ]; then
    echo "Cloning GEAK-agent..."
    git clone https://github.com/AMD-AGI/GEAK-agent.git GEAK-agent
else
    echo "GEAK-agent already exists, skipping clone"
fi

# 2. Clone GEAK-eval (openevolve branch for tb_eval)
if [ ! -d "GEAK-eval" ]; then
    echo "Cloning GEAK-eval (openevolve branch)..."
    git clone --branch openevolve https://github.com/AMD-AGI/GEAK-eval.git GEAK-eval
else
    echo "GEAK-eval already exists, skipping clone"
fi

# 3. Install GEAK-agent requirements (excluding torch - assumes ROCm torch is pre-installed)
if [ -f "GEAK-agent/requirements.txt" ]; then
    echo "Installing GEAK-agent requirements..."
    pip install -r GEAK-agent/requirements.txt --ignore-installed torch torchvision torchaudio 2>/dev/null || true
fi

# 4. Install GEAK-eval as editable package
if [ -d "GEAK-eval" ]; then
    echo "Installing GEAK-eval (tb_eval)..."
    cd GEAK-eval
    pip install -e . --no-deps
    cd ..
fi

echo "=============================================="
echo "Setup complete!"
echo "=============================================="

