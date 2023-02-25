# Note: When rerunning the build, you can save time by:
# docker build --cache-from visual-perma-tracker:latest -t visual-perma-trackere:latest .
FROM python:3.10

WORKDIR /home

COPY requirements.txt /home/requirements.txt

# Install OS dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx cmake

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install setuptools \
    && pip install --default-timeout=10000 Cython \
    && pip install numpy \
    && pip install --default-timeout=10000 dlib \
    && pip install --default-timeout=10000000 -r requirements.txt

# Create directories
RUN mkdir /home/data /home/data/images /home/data/database /home/data/identities

# Copy files
COPY /src /home/src
COPY /external /home/external
COPY /model /home/model
COPY /configs /home/configs
COPY main.py /home/main.py

# Build the bytetrack package
RUN cd /home/external/bytetrack && python3 setup.py -q develop

# Set the default command
CMD ["sh", "-c", "python main.py $@ /home/data/identities /home/data /home/configs"]