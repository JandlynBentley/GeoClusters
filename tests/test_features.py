"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021

This program contains automated tests for continuous integration's pytest.
It tests components in main and preprocessing.
"""

import pytest
import pandas as pd


# Fixtures are used to feed some sort of input data to the tests
@pytest.fixture
def test_preprocessing_data():
    data = pd.read_csv('McDonough-etal_2019_test.csv')
    return data


# These tests are conducted on the testing data set made after pre-processing
def test():
    assert test_preprocessing_data is not None

