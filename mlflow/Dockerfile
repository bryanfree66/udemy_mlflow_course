FROM continuumio/miniconda3:4.9.2-alpine
RUN /opt/conda/bin/conda install -c conda-forge mlflow==1.18.0 pymysql=1.0.2 boto3==1.17.101 -y --freeze-installed && conda clean -afy