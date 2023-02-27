# Note: When rerunning the build, you can save time by:
# DOCKER_BUILDKIT=1 docker build --cache-from visual-perma-tracker:latest -t visual-perma-tracker:latest .
# sudo docker run --runtime=nvidia --gpus all -v /home/moritz/Workspace/masterthesis/data:/home/data -v /home/moritz/Workspace/masterthesis/data/identities:/home/data/identities -v /home/moritz/Workspace/masterthesis/configs:/home/configs mo12896/visual-perma-tracker:0.0.6
# Enter the container after running: docker run --rm -it --entrypoint=/bin/bash mo12896/visual-perma-tracker:0.0.7
FROM python:3.10

WORKDIR /home

COPY requirements.txt /home/requirements.txt

# Install OS dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx cmake protobuf-compiler

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 \
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

# Set the default command
CMD ["sh", "-c", "python main.py $@"]