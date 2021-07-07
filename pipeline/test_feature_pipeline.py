import pytest
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from .feature_pipeline import impute_missing, create_feature_vector, scale_features

@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[*]").appName("test").getOrCreate()

#@pytest.mark.usefixtures("spark_session")
def test_imputation_of_missing_data(spark_session):
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

    input_cols = ['Column1','Column2', 'Column3']
    output_cols = ['Column1_imp', 'Column2_imp', 'Column3_imp']

    # Call method under test
    result_df = impute_missing(df, input_cols, output_cols)
    # Get null counts in output columns
    null_count_1 = result_df.filter(col('Column1_imp').isNull()).count()
    null_count_2 = result_df.filter(col('Column2_imp').isNull()).count()
    null_count_3 = result_df.filter(col('Column3_imp').isNull()).count()
    # Make Assertions
    assert isinstance(result_df, DataFrame)
    assert ['Column1','Column2', 'Column3', 'Label', 'Column1_imp', 'Column2_imp', 'Column3_imp'] == result_df.columns
    assert 0 == null_count_1
    assert 0 == null_count_2
    assert 0 == null_count_3

#@pytest.mark.usefixtures("spark_session")
def test_create_feature_vector(spark_session):
    df = spark_session.createDataFrame(
        [
            (98.3679148956603,28415.57583214058,10.558949998467961,1),
            (103.46475866009455,27420.16742458204,8.417305032089528,0),
            (108.91662923953173,14476.335695268315,5.398162017711099,1),
            (113.17596460727073,9943.92978526269,6.337137942441213,1),
            (114.7335449715346,13677.99404000127,9.981200455815905,1)
        ],
        ['Column1','Column2','Column3','Label']
    )
    input_cols = ['Column1','Column2','Column3',]
    output_col='Features'

    # Call method under test
    result_df = create_feature_vector(df, input_cols, output_col)
    # Make Assertions
    assert isinstance(result_df, DataFrame)
    assert ['Column1','Column2','Column3', 'Label', 'Features'] == result_df.columns
    assert 3 == result_df.schema["Features"].metadata["ml_attr"]["num_attrs"]

    #@pytest.mark.usefixtures("spark_session")
def test_scale_features(spark_session):
    df = spark_session.createDataFrame(
        [
            ((Vectors.dense([98.3679, 28415.5758, 10.5589])),1),
            ((Vectors.dense([103.4648, 27420.1674, 8.4173])),0),
            ((Vectors.dense([108.9166, 14476.3357, 5.3982])),1)
        ],
        ['Features', 'Label']
    )
    input_col = 'Features'
    output_col='ScaledFeatures'

    # Call method under test
    result_df = scale_features(df, input_col, output_col)
    
    # Make Assertions
    assert isinstance(result_df, DataFrame)
    assert ['Features', 'Label', 'ScaledFeatures'] == result_df.columns
    assert 3 == result_df.schema["ScaledFeatures"].metadata["ml_attr"]["num_attrs"]