# Note: When rerunning the build, you can save time by:
# DOCKER_BUILDKIT=1 docker build --cache-from visual-perma-tracker:latest -t visual-perma-tracker:latest .
# DOCKER_BUILDKIT=1 docker build -t mo12896/visual-perma-tracker:0.0.8 .
# sudo docker run --gpus all -v /home/moritz/Workspace/masterthesis/data:/home/data -v /home/moritz/Workspace/masterthesis/data/identities:/home/data/identities -v /home/moritz/Workspace/masterthesis/configs:/home/configs mo12896/visual-perma-tracker:0.0.8
# Enter the container after running: docker run --rm -it --entrypoint=/bin/bash mo12896/visual-perma-tracker:0.0.8
# Worked, but issues with CUDA
FROM nvidia/cuda:11.5.0-base-ubuntu20.04

WORKDIR /home

COPY requirements.txt /home/requirements.txt

# Add the deadsnakes PPA for Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common libgl1-mesa-glx cmake protobuf-compiler && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and dev packages
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip  && \
    rm -rf /var/lib/apt/lists/*

# Install virtualenv
RUN pip3 install virtualenv

# Create a virtual environment with Python 3.10
RUN virtualenv -p python3.10 venv

# Activate the virtual environment
ENV PATH="/home/venv/bin:$PATH"

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip3 install --default-timeout=10000000 -r requirements.txt

# Create directories
RUN mkdir /home/data /home/data/images /home/data/database /home/data/identities /home/logs

# Copy files
COPY /src /home/src
COPY res10_300x300_ssd_iter_140000.caffemodel /home/res10_300x300_ssd_iter_140000.caffemodel
COPY /external /home/external
COPY /model /home/model
COPY /configs /home/configs
COPY main.py /home/main.py

# Build the bytetrack package
RUN cd /home/external/bytetrack && python3 setup.py -q develop
#New, but should still work:
RUN cd /home/external/synergy/Sim3DR && ./build_sim3dr.sh
RUN cd /home/external/synergy/FaceBoxes && ./build_cpu_nms.sh

# Tested with only cuda, not with cuda-11.5, yet
# Set the PYTHONPATH and LD_LIBRARY_PATH environment variable to include the CUDA libraries
ENV PYTHONPATH=/usr/local/cuda-11.5/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64

# Set the CUDA_PATH and CUDA_HOME environment variable to point to the CUDA installation directory
ENV CUDA_PATH=/usr/local/cuda-11.5
ENV CUDA_HOME=/usr/local/cuda-11.5

# Set the default command
CMD ["sh", "-c", ". /home/venv/bin/activate && python main.py $@"]


# The first version which gets python3.10 through!
# Start with the nvidia/cuda:11.5.0-base-ubuntu20.04 image as the base image
# FROM nvidia/cuda:11.5.0-base-ubuntu20.04

# WORKDIR /home

# COPY requirements.txt /home/requirements.txt

# # Add the deadsnakes PPA for Python 3.10
# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update

# # Install Python 3.10 and dev packages
# RUN apt-get update && \
#     apt-get install -y python3.10 python3.10-dev python3-pip  && \
#     rm -rf /var/lib/apt/lists/*

# # RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# # RUN update-alternatives --set python3 /usr/bin/python3.10

# # Install Python dependencies
# RUN pip3 install --upgrade pip \
#     && pip3 install --default-timeout=10000000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
#     && pip3 install --default-timeout=10000000 -r requirements.txt

# # Create directories
# RUN mkdir /home/data /home/data/images /home/data/database /home/data/identities /home/logs

# # Copy files
# COPY /src /home/src
# COPY res10_300x300_ssd_iter_140000.caffemodel /home/res10_300x300_ssd_iter_140000.caffemodel
# COPY /external /home/external
# COPY /model /home/model
# COPY /configs /home/configs
# COPY main.py /home/main.py

# # Build the bytetrack package
# RUN cd /home/external/bytetrack && python3 setup.py -q develop
# #New, but should still work:
# RUN cd /home/external/synergy/Sim3DR && ./build_sim3dr.sh
# RUN cd /home/external/synergy/FaceBoxes && ./build_cpu_nms.sh

# # Set the PYTHONPATH and LD_LIBRARY_PATH environment variable to include the CUDA libraries
# ENV PYTHONPATH=/usr/local/cuda/lib64
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# # Set the CUDA_PATH and CUDA_HOME environment variable to point to the CUDA installation directory
# ENV CUDA_PATH=/usr/local/cuda
# ENV CUDA_HOME=/usr/local/cuda

# # Set the default command
# CMD ["sh", "-c", "python3.10 main.py $@"]



#Simple setup
# FROM python:3.10

# WORKDIR /home

# COPY requirements.txt /home/requirements.txt

# # Install OS dependencies
# RUN apt-get update && apt-get install -y libgl1-mesa-glx cmake protobuf-compiler

# # Install Python dependencies
# RUN pip3 install --upgrade pip \
#     && pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
#     && pip3 install --default-timeout=10000000 -r requirements.txt

# # Create directories
# RUN mkdir /home/data /home/data/images /home/data/database /home/data/identities /home/logs

# # Copy files
# COPY /src /home/src
# COPY res10_300x300_ssd_iter_140000.caffemodel /home/res10_300x300_ssd_iter_140000.caffemodel
# COPY /external /home/external
# COPY /model /home/model
# COPY /configs /home/configs
# COPY main.py /home/main.py

# # Build the bytetrack package
# RUN cd /home/external/bytetrack && python3 setup.py -q develop
# RUN cd /home/external/synergy/Sim3DR && ./build_sim3dr.sh
# RUN cd /home/external/synergy/FaceBoxes && ./build_cpu_nms.sh

# # Set the default command
# CMD ["sh", "-c", "python main.py $@"]



