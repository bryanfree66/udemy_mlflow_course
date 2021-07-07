import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer, VectorAssembler, PolynomialExpansion, StandardScaler

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
Creates a feature vector column in the Spark DataFrame with values in input columns.
Outputs a Spark DataFrame with output column appended
"""
def create_feature_vector(df, input_cols, output_col) -> DataFrame:
    vec_assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol=output_col
    )
    return vec_assembler.transform(df)

"""Scales the features of a Dataframe
Creates a new scaled features column in a Dataframe from a column of feature vectors
Outputs a Spark DataFrame with the output column appended
"""
def scale_features(df, input_col, output_col) -> DataFrame:
    scaler = StandardScaler(
        inputCol=input_col, outputCol=output_col,
        withStd=True, withMean=True
    )
    scaler_fit = scaler.fit(df)
    return scaler_fit.transform(df)