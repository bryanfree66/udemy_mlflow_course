import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def load_dataset(spark, file_path) -> DataFrame:
    return spark.read.csv(file_path,inferSchema=True, header=True)

