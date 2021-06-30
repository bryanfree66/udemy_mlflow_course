import pytest
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from .feature_pipeline import load_dataset

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
    return spark

#@pytest.mark.usefixtures("spark_session")
def test_something(spark_session):
    test_df = spark_session.createDataFrame(
        [
            (1,None,103.46475866009455,27420.16742458204,8.417305032089528,None,485.97450045781375,11.351132730708514,67.8699636759021,4.620793451653219,0),
            (3,None,113.17596460727073,9943.92978526269,6.337137942441213,354.29756524708256,415.3383368798727,19.67616854859483,23.07580599653685,3.787475537347365,1),
            (23,None,154.92597170844743,30037.221624651935,5.796570971275747,252.06726719561706,311.7500192169767,13.904652296228505,77.53204198223717,4.1135850300896575,1),
            (367,0.22749905020219874,152.5301111764229,39028.599340290755,3.4624920476792767,283.69378223429663,443.0292321286284,13.201943203829217,62.32271110691731,3.545741437567914,1),
            (434,4.126528715100222,125.47488351387379,11215.945901592888,5.366011335667973,261.44479767930693,445.2414568851872,18.575920677608508,86.43365536398535,4.4602009837552865,0)
        ],
        ['', 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',  'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
    )
    # CALL METHOD
    # Make Assertion
    assert isinstance(test_df, DataFrame)

