import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer, VectorAssembler, PolynomialExpansion, StandardScaler

"""Load the Dataset
Loads a CSV formatted dataset with header from the specified file path.
Outputs a Spark DataFrame.
"""
def load_dataset(spark, file_path) -> DataFrame:
    return spark.read.csv(file_path,inferSchema=True, header=True)

"""Imputes missing values in dataset
Imputes missing values input columns based on the mean of the column.
Outputs a Spark DataFrame with missing columns appended.
"""
def impute_missing(df, input_cols, output_cols) -> DataFrame:
    imputer = Imputer(
        inputCols=input_cols,
        outputCols=output_cols
    )
    return imputer.setStrategy("mean").fit(df).transform(df)

"""Creates feature vector column in DataFrame
Creates a feature vector column in the Spark Dataframe with values in input columns.
Outputs a Spark DataFrame with output column appended
"""
def create_feature_vector(df, input_cols, output_col) -> DataFrame:
    vec_assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol=output_col
    )
    return vec_assembler.transform(df)