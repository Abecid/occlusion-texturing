#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Check if git-lfs is installed; if not, install it
if ! command -v git-lfs &> /dev/null
then
    echo "git-lfs could not be found, installing..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
fi

# Initialize git-lfs
git lfs install

# Clone the repository from Hugging Face
git clone https://huggingface.co/h94/IP-Adapter

# Move the necessary directories
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models

echo "Setup complete!"
