#!/bin/zsh
# Script to activate the risk-tool conda environment

# Source conda
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh

# Activate the risk-tool environment
conda activate risk-tool

# Confirm activation
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
