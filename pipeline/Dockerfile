
FROM alpine:3.14.0 as download
RUN apk add --no-cache wget
RUN wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2/ /usr/local/spark

FROM openjdk:8-alpine
ENV PYSPARK_PYTHON=python3
COPY --from=download /usr/local/spark /usr/local/spark
RUN apk add --no-cache bash python3 python3-dev py3-pip gfortran linux-headers build-base musl-dev g++ && \
    pip3 install cython numpy pytest pyspark && \
    ln -s /usr/local/spark spark
WORKDIR /app
