#!/bin/bash
# Setup script for NVIDIA Container Toolkit on Ubuntu/Lambda Labs
# This enables GPU support in Docker containers
# Based on: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

set -e

echo "=== Installing NVIDIA Container Toolkit ==="
echo ""
echo "Prerequisites:"
echo "1. Ensure NVIDIA GPU driver is installed"
echo "2. Ensure Docker is installed"
echo ""

# Check if running on Ubuntu/Debian
if [ ! -f /etc/os-release ]; then
    echo "Error: Cannot detect OS. This script is for Ubuntu/Debian."
    exit 1
fi

# Install prerequisites
echo "=== Step 1: Installing prerequisites ==="
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    curl \
    gnupg2

# Configure the production repository
echo ""
echo "=== Step 2: Configuring NVIDIA repository ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list
echo ""
echo "=== Step 3: Updating package list ==="
sudo apt-get update

# Install the NVIDIA Container Toolkit packages
echo ""
echo "=== Step 4: Installing NVIDIA Container Toolkit ==="
# Install latest version (or specify version like: export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1)
sudo apt-get install -y \
    nvidia-container-toolkit \
    nvidia-container-toolkit-base \
    libnvidia-container-tools \
    libnvidia-container1

# Configure Docker to use NVIDIA runtime
echo ""
echo "=== Step 5: Configuring Docker runtime ==="
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo ""
echo "=== Step 6: Restarting Docker daemon ==="
sudo systemctl restart docker

echo ""
echo "=== Installation complete! ==="
echo ""
echo "=== Verifying installation ==="
echo "Running test container with GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu24.04 nvidia-smi; then
    echo ""
    echo "✓ GPU access verified successfully!"
    echo "You can now use --gpus all in your docker run commands."
else
    echo ""
    echo "⚠ Warning: GPU verification failed. Please check:"
    echo "  1. NVIDIA GPU driver is installed (run: nvidia-smi)"
    echo "  2. Docker daemon restarted successfully"
    echo "  3. You have appropriate permissions"
    exit 1
fi

