import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer, VectorAssembler, PolynomialExpansion, StandardScaler

def load_dataset(spark, file_path) -> DataFrame:
    return spark.read.csv(file_path,inferSchema=True, header=True)

def impute_missing(df, input_cols, output_cols) -> DataFrame:
    imputer = Imputer(
        inputCols=input_cols,
        outputCols=output_cols
    )
    return imputer.setStrategy("mean").fit(df).transform(df)

def create_feature_vector(df, input_cols, output_col) -> DataFrame:
    vec_assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol=output_col
    )
    return vec_assembler.transform(df)