FROM python:3.10

WORKDIR /home

COPY requirements.txt /home/requirements.txt

# This takes a while, since we have to install a lot of large dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade setuptools \
    && pip install --default-timeout=10000 Cython \
    && pip install numpy==1.21.6 \
    && pip install --default-timeout=10000000 -r requirements.txt \
    && mkdir /home/data /home/data/images /home/data/database /home/data/identities

COPY /src /home/src
COPY /external /home/external
COPY /model /home/model
COPY /configs /home/configs
COPY main.py /home/main.py
COPY create_embedding_database.py /home/create_embedding_database.py

CMD ["sh", "-c", "python create_embedding_database.py $@ /home/data /home/data/images & python main.py $@ /home/data/identities /home/data"]