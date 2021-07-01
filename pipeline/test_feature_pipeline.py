import pytest
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import isnan, when, count, col
from .feature_pipeline import impute_missing, create_feature_vector

@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[*]").appName("test").getOrCreate()

#@pytest.mark.usefixtures("spark_session")
def test_imputation_of_missing_data(spark_session):
    df = spark_session.createDataFrame(
        [
            (1,None,103.46475866009455,27420.16742458204,8.417305032089528,None,485.97450045781375,11.351132730708514,67.8699636759021,4.620793451653219,0),
            (3,None,113.17596460727073,9943.92978526269,6.337137942441213,354.29756524708256,415.3383368798727,19.67616854859483,None,3.787475537347365,1),
            (23,None,154.92597170844743,30037.221624651935,5.796570971275747,252.06726719561706,311.7500192169767,13.904652296228505,77.53204198223717,4.1135850300896575,1),
            (367,0.22749905020219874,152.5301111764229,39028.599340290755,3.4624920476792767,283.69378223429663,443.0292321286284,13.201943203829217,62.32271110691731,3.545741437567914,1),
            (434,4.126528715100222,125.47488351387379,11215.945901592888,5.366011335667973,261.44479767930693,445.2414568851872,18.575920677608508,86.43365536398535,4.4602009837552865,0)
        ],
        ['', 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',  'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
    )

    input_cols = ['ph', 'Sulfate', 'Trihalomethanes']
    output_cols = ['ph_imp', 'Sulfate_imp', 'Trihalomethanes_imp']

    # Call method under test
    result_df = impute_missing(df, input_cols, output_cols)
    # Get null counts in output columns
    ph_imp_null_count = result_df.filter(col('ph_imp').isNull()).count()
    sulfate_imp_null_count = result_df.filter(col('Sulfate_imp').isNull()).count()
    trihalomethanes_imp_null_count = result_df.filter(col('Trihalomethanes_imp').isNull()).count()
    # Make Assertions
    assert isinstance(result_df, DataFrame)
    assert 0 == ph_imp_null_count
    assert 0 == sulfate_imp_null_count
    assert 0 == trihalomethanes_imp_null_count

#@pytest.mark.usefixtures("spark_session")
def test_create_feature_vector(spark_session):
    df = spark_session.createDataFrame(
        [
            (98.3679148956603,28415.57583214058,10.558949998467961,505.24026927891407,12.882614472289333,4.119087300328971,7.065394544064872,296.843207792478,85.32995534051292,1),
            (103.46475866009455,27420.16742458204,8.417305032089528,485.97450045781375,11.351132730708514,4.620793451653219,7.065394544064872,333.8051408041043,67.8699636759021,0),
            (108.91662923953173,14476.335695268315,5.398162017711099,512.2323064106689,15.013793389990155,3.895572062268123,7.065394544064872,281.198274407849,86.6714587149138,1),
            (113.17596460727073,9943.92978526269,6.337137942441213,415.3383368798727,19.67616854859483,3.787475537347365,7.065394544064872,354.29756524708256,23.07580599653685,1),
            (114.7335449715346,13677.99404000127,9.981200455815905,524.000355172102,11.384858471731945,3.2938483740192734,7.065394544064872,441.82677662870003,71.15328465919002,1)
        ],
        ['Hardness','Solids','Chloramines','Conductivity','Organic_carbon','Turbidity','ph_imp', 'Sulfate_imp', 'Trihalomethanes_imp']
    )
    input_cols = ['Hardness','Solids','Chloramines','Conductivity','Organic_carbon','Turbidity','ph_imp', 'Sulfate_imp', 'Trihalomethanes_imp']
    output_col='Features'

    # Call method under test
    result_df = create_feature_vector(df, input_cols, output_col)
    row = result_df.select('Features').take(2)
    row_dict_0 = row[0].asDict()
    vector_0 = row_dict_0['Features']
    row_dict_1 = row[1].asDict()
    vector_1 = row_dict_1['Features']
    # Make Assertions
    assert isinstance(result_df, DataFrame)
    assert 98.3679148956603 == vector_0[0]
    assert 103.46475866009455 == vector_1[0]