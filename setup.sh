#!/bin/bash
mkdir visual-perma-tracker && cd visual-perma-tracker
git clone https://github.com/mo12896/emotion-recognition.git

# Add the deadsnakes PPA for Python 3.10
sudo apt-get update && \
sudo apt-get install -y libgl1-mesa-glx cmake protobuf-compiler && \
sudo apt-get update

# Create a Conda environment with Python 3.10
conda create -n venv python=3.10

# Activate the Conda environment
conda activate venv

# Install Python dependencies
pip install --upgrade pip \
    && pip install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip install --default-timeout=10000000 -r requirements.txt

# Create directories
mkdir /home/data /home/data/images /home/data/database /home/data/identities /home/logs

# Copy files
# Example:
# cp source/path destination/path

# Build the bytetrack package
cd /home/external/bytetrack && python setup.py -q develop
#New, but should still work:
cd /home/external/synergy/Sim3DR && ./build_sim3dr.sh
cd /home/external/synergy/FaceBoxes && ./build_cpu_nms.sh