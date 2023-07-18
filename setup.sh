#!/bin/bash

echo "Virtual Environment: Enter 1 to use conda, 2 to use venv: "
read choice

# Setting up the working directory and creating a virtual environment
if [ $choice -eq 1 ]
then
    echo "Setting up conda environment..."
    cd facial-analysis-system
    conda create -n facesys python=3.8
    conda activate facesys
elif [ $choice -eq 2 ]
then
    echo "Setting up venv environment..."
    cd facial-analysis-system
    python3.8 -m venv emorec
    source emorec/bin/activate
else
    echo "Invalid choice, exiting setup script..."
    exit 1
fi

# Installing OS tools
echo "Installing OS tools..."
sudo apt-get update && \
sudo apt-get install -y libgl1-mesa-glx cmake protobuf-compiler

# Installing required packages
echo "Installing required packages..."
pip3 install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -p -r requirements.txt

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



