from typing import List
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, PolynomialExpansion, StandardScaler

def resample(df,ratio,target, majority_value):
    positive = df.filter(F.col(target)==majority_value)
    negative = df.filter(F.col(target)!=majority_value)
    total_positive = positive.count()
    total_negative = negative.count()
    fraction=float(total_positive * ratio)/float(total_negative)
    sampled = negative.sample(False,fraction)
    return sampled.union(positive)

def create_imputer(input_cols:List, output_cols:List) -> Imputer:
    """Creates an Imputer object that accept List of columns and outputs List of columns"""
    return Imputer(
        inputCols=input_cols,
        outputCols=output_cols
    ).setStrategy("mean")

def create_assembler(input_cols:List, output_col:str) -> VectorAssembler:
    """Creates Vector assembler object accepts List of columns and outputs feature vector column"""
    return VectorAssembler(
        inputCols=input_cols,
        outputCol=output_col
    )

def create_scaler(input_col:str, output_col:str) -> StandardScaler:
    """Create StandardScaler object to scale the feature vector"""
    return StandardScaler(
        inputCol=input_col,
        outputCol=output_col,
        withStd=True,
        withMean=True
    )

def create_expander(input_col:str, output_col:str, exp_degree:int) -> PolynomialExpansion:
    """Creates PolynomialExpansion object to expand the scaled features in polynomial space"""
    return PolynomialExpansion(degree=exp_degree,
        inputCol=input_col,
        outputCol=output_col
    )

def create_feature_pipeline(feature_cols: List, assembler_out_col:str, scaler_out_col:str, expander_out_col:str, degree:int) -> Pipeline:
    """Create a new feature pipeline and fit to the training data
    Creates and returns a pipeline model from from the individual pipeline stages 
    """
    imputer = create_imputer(input_cols=feature_cols, output_cols=feature_cols)
    assembler = create_assembler(input_cols=feature_cols, output_col=assembler_out_col)
    scaler = create_scaler(input_col=assembler_out_col, output_col=scaler_out_col)
    expander = create_expander(input_col=scaler_out_col, output_col=expander_out_col, exp_degree=degree)
    return Pipeline(stages=[imputer, assembler, scaler, expander])
