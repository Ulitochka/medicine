FROM phusion/baseimage:0.9.19

RUN locale-gen ru_RU.UTF-8
ENV LANG ru_RU.UTF-8

MAINTAINER m.domrachev.scientist@gmail.com

ENV BUILD_THREADS 4

RUN apt-get update && \
    apt-get --no-install-recommends --no-upgrade -y --force-yes install \
    mc \
    python3.5 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libpython3.5 \
    libboost-python1.58.0 \
    python3-tk && \
    pip3 install --upgrade pip && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean

WORKDIR /code

COPY . /code

RUN pip3 install -r requirements-py3.txt && \
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

CMD ["/bin/bash"]