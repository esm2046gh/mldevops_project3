# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:52:53 2022

@author: esm
"""
#%% pandas_profiling
import pandas as pd

from data import process_data
#import pandas_profiling

local_path = '../../data/census_clean.csv'
df = pd.read_csv(local_path)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_data, y_data, encoder, lb = process_data(
    df, categorical_features=cat_features, label="salary", training=True
)

#profile = pandas_profiling.ProfileReport(X_data)
#profile.to_notebook_iframe()
#profile.to_widgets()

#%% 5. Unit Testing for Slice-Validation
import pandas as pd
import pytest


#@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "numeric_feat": [3.14, 2.72, 1.62],
            "categorical_feat": ["dog", "dog", "cat"],
        }
    )
    return df


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_slice_averages(data):
    """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
    for cat_feat in data["categorical_feat"].unique():
        avg_value = data[data["categorical_feat"] == cat_feat]["numeric_feat"].mean()
        assert (
            2.5 > avg_value > 1.5
        ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."
        
#%% slice_iris
#sepal_length,sepal_width,petal_length,petal_width,species
import pandas as pd

df = pd.read_csv("C:\\ml\\Anaconda3v1\\Lib\\site-packages\\bokeh\sampledata\\_data\\iris.csv")


def slice_iris(df, feature):
    """ Function for calculating descriptive stats on slices of the Iris dataset."""
    for class0 in df["species"].unique():
        df_temp = df[df["species"] == class0]
        mean = df_temp[feature].mean()
        stddev = df_temp[feature].std()
        print(f"Class: {class0}")
        print(f"{feature} mean: {mean:.4f}")
        print(f"{feature} stddev: {stddev:.4f}")
    print()

if __name__ == "__main__":
        
    slice_iris(df, "sepal_length")
    slice_iris(df, "sepal_width")
    slice_iris(df, "petal_length")
    slice_iris(df, "petal_width")