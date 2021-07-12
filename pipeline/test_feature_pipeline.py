import pytest
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from .feature_pipeline import create_expander, create_imputer, create_scaler, create_assembler, create_feature_pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, PolynomialExpansion, StandardScaler


@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[*]").appName("test").getOrCreate()

#@pytest.mark.usefixtures("spark_session")
def test_create_imputer(spark_session):
    input_cols = ['Column1','Column2', 'Column3']
    output_cols = ['Column1_imp', 'Column2_imp', 'Column3_imp']
    result = create_imputer(input_cols, output_cols)
    assert isinstance(result, Imputer)
    assert input_cols  == result.getInputCols()
    assert output_cols  == result.getOutputCols()

#@pytest.mark.usefixtures("spark_session")
def test_create_assembler(spark_session):
    input_cols = ['Column1','Column2', 'Column3']
    output_col = 'Features'
    result = create_assembler(input_cols, output_col)
    assert isinstance(result, VectorAssembler)
    assert input_cols  == result.getInputCols()
    assert output_col  == result.getOutputCol()

#@pytest.mark.usefixtures("spark_session")
def test_create_scaler(spark_session):
    input_col = 'Features'
    output_col = 'ScaledFeatures'
    result = create_scaler(input_col, output_col)
    assert isinstance(result, StandardScaler)
    assert input_col  == result.getInputCol()
    assert output_col  == result.getOutputCol()

#@pytest.mark.usefixtures("spark_session")
def test_create_expander(spark_session):
    input_col = 'ScaledFeatures'
    output_col = 'ExpandedFeatures'
    degree = 3
    result = create_expander(input_col, output_col, degree)
    assert isinstance(result, PolynomialExpansion)
    assert input_col  == result.getInputCol()
    assert output_col  == result.getOutputCol()

#@pytest.mark.usefixtures("spark_session")
def test_create_feature_pipeline(spark_session):
    df = spark_session.createDataFrame(
        [
            (None,8.417305032089528,None,0),
            (None,6.337137942441213,354.29756524708256,1),
            (None,None,252.06726719561706,1),
            (0.22749905020219874,3.4624920476792767,283.693782234296631,1),
            (4.126528715100222,5.366011335667973,None,0)
        ],
        ['Column1','Column2', 'Column3', 'Label']
    )

    feature_cols = ['Column1','Column2', 'Column3']
    assembler_out_col = 'Features'
    scaler_out_col = 'ScaledFeatures'
    expander_out_col = 'ExpandedFeatures'
    degree = 3
    result = create_feature_pipeline(df, feature_cols, assembler_out_col, scaler_out_col, expander_out_col, degree)
    assert isinstance(result, PipelineModel)
