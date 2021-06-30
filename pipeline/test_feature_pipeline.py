import pytest
from .feature_pipeline import load_dataset

def test_load_dataset_result_is_not_None():
    result = load_dataset('/data/water_potability_test.csv')
    assert result is not None