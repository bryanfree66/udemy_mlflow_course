import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer

def load_dataset(spark, file_path) -> DataFrame:
    return spark.read.csv(file_path,inferSchema=True, header=True)

def impute_missing(df, input_cols, output_cols) -> DataFrame:
    imputer = Imputer(
        inputCols=input_cols,
        outputCols=output_cols
    )
    return imputer.setStrategy("mean").fit(df).transform(df)