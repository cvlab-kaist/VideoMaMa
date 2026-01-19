#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# Create conda environment
echo "Creating conda environment..."

echo "👉 Setting up SAM 2 environment..."
conda create -n sam2 python=3.11 -y
conda activate sam2

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install -e ".[dev]"

conda create -n videomama python=3.9 -y
conda activate videomama

# Install pytorch
echo "Installing pytorch..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies and local package
echo "Installing local package..."
pip install psutil
pip install -e . --no-build-isolation

# install latest huggingface for dino
# pip install git+https://github.com/huggingface/transformers
# 4.57 not released yet.
pip install transformers==4.57.0

# Git-LFS
echo "👉 Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs

# Git LFS reset
echo "👉 Initializing Git LFS..."

git lfs install

# Download SVD
mkdir checkpoints
cd checkpoints

git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
git clone https://huggingface.co/SammyLim/VideoMaMa

mkdir sam2
cd sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ~/

echo "✅ Git LFS installation complete!"

echo "Environment setup complete!"