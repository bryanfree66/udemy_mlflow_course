FROM ubuntu:21.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y -q default-jdk scala software-properties-common python3.9 python3-pip libpq-dev build-essential libssl-dev libffi-dev python3-dev unzip wget \
    && apt-get clean

RUN wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2/ /usr/local/spark && \
    ln -s /usr/local/spark spark

WORKDIR /app

COPY . /app

RUN pip3 install cython==0.29.21 numpy==1.18.5 pytest pyspark

ENV PYSPARK_PYTHON=python3