
FROM openjdk:8-alpine

ENV PYSPARK_PYTHON=python3

RUN apk add --no-cache bash python3 python3-dev py3-pip gfortran build-base musl-dev linux-headers g++ wget 
RUN pip3 install cython numpy pytest pyspark
RUN wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    rm spark-3.1.2-bin-hadoop3.2.tgz && \ 
    mv spark-3.1.2-bin-hadoop3.2/ /usr/local/spark && \
    ln -s /usr/local/spark spark

WORKDIR /app
