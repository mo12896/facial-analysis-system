#!/bin/bash

# Install OS dependencies
sudo apt-get update && \
sudo apt-get install -y libgl1-mesa-glx cmake protobuf-compiler git

# Download the Conda installer script
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Clone the repository
git clone https://github.com/mo12896/emotion-recognition.git
cd emotion-recognition

# Create a Conda environment with Python 3.10
conda create -n venv python=3.10
conda activate venv

# Install Python dependencies
pip3 install --upgrade pip \
    && pip3 install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip3 install --default-timeout=10000000 -r requirements.txt

# Create directories
mkdir /data /data/images /data/database /data/identities /logs /external

# Download the model
git lfs pull https://huggingface.co/liangtian/birthdayCrown/raw/main/res10_300x300_ssd_iter_140000.caffemodel

# Setup the git submodules
git submodule init
git submodule update

# Build the bytetrack package
#cd /home/external/bytetrack && python setup.py -q develop
#New, but should still work:
# Define variables
FILE_ID="1BVHbiLTfX6iTeJcNbh-jgHjWDoemfrzG"
FILE_NAME="best.pth.tar"
FOLDER_NAME="./external/synergy/pretrained"
# Download the file to the specified folder
curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o "${FOLDER_NAME}/${FILE_NAME}"


# Define variables
FILE_ID="1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8"
FILE_NAME="3dmm_data.zip"
FOLDER_NAME="./external/synergy/"
# Download the file to the specified folder
curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o "${FOLDER_NAME}/${FILE_NAME}"
unzip "${FOLDER_NAME}/${FILE_NAME}" -d "${FOLDER_NAME}"

cd /external/synergy/Sim3DR && ./build_sim3dr.sh
cd /external/synergy/FaceBoxes && ./build_cpu_nms.sh
cd external/synergy/ && pip install -e .


