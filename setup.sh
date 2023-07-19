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
cd external/synergy/pretrained/

gdown 'https://drive.google.com/uc?id=1BVHbiLTfX6iTeJcNbh-jgHjWDoemfrzG' -O best.pth.tar

cd ..
gdown 'https://drive.google.com/uc?id=1YVBRcXmCeO1t5Bepv67KVr_QKcOur3Yy' -O aflw2000_data.zip
gdown 'https://drive.google.com/uc?id=1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8' -O 3dmm_data.zip

unzip aflw2000_data.zip -d aflw2000_data
unzip 3dmm_data.zip -d 3dmm_data

echo "Check, if best.pth.tar was fetched into external/synergy/pretrained and aflw2000_data and 3dmm_data where fetched and unzipped into external/synergy. Then press Enter to continue..."
read

cd Sim3DR
chmod +x build_sim3dr.sh
./build_sim3dr.sh

cd ../FaceBoxes
chmod +x build_cpu_nms.sh
./build_cpu_nms.sh

cd ../../..

# Setting up the directory structure
echo "Setting up the directory structure..."
mkdir logs data data/output data/input

echo "Setup completed."


