#!/bin/bash

# Installing OS tools
echo "Installing OS tools..."
sudo apt-get update && \
sudo apt-get install -y libgl1-mesa-glx cmake protobuf-compiler

# Installing required packages
echo "Installing required packages..."
pip3 install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirements.txt

# Setting up the SynergyNet API
echo "Setting up the SynergyNet API..."
cd external/synergy/Sim3DR/

chmod +x build_sim3dr.sh
./build_sim3dr.sh

cd ../FaceBoxes

chmod +x build_cpu_nms.sh
./build_cpu_nms.sh

cd ../../..

# Setting up the directory structure
echo "Setting up the directory structure..."
mkdir data data/output data/input

echo "Setup completed."



