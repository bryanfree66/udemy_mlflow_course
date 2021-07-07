import pyspark
from pyspark.sql import SparkSession, DataFrame

"""Get or Create a Spark SQL session and add support for MLeap."""
def get_spark_session() -> SparkSession:
    return (SparkSession.builder
        .config("spark.jars.packages","ml.combust.mleap:mleap-spark-base_2.11:0.17.0,ml.combust.mleap:mleap-spark_2.11:0.17.0")
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .getOrCreate()
    )
    
""" Loads and returns a Spark DataFrame from a CSV file at the specified path"""
def load_data(spark_session, file_path) -> DataFrame:
    return spark_session.read.csv(file_path,inferSchema=True, header=True)

