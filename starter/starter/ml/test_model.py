"""
- Module to test model.py
- This is part of Project-3 of Machine Learning DevOps Nano Degree

Author: E. Saavedra
Date: Jan. 2022
"""

import math
from random import random
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from starter.starter.ml import model
import model


#@pytest.fixture
def test_compute_model_metrics():
    #Arrange
    y = [1, 0, 0]
    y_preds = [1, 1, 0]

    #Act
    metrics = model.compute_model_metrics(y, y_preds)

    #Asserts
    assert math.isclose(metrics.fbeta, 0.6666, rel_tol=1e-04), "fbeta should be 0.6666"
    assert metrics.precision == 0.5, "precision should be 0.5"
    assert metrics.recall == 1.0, "recall should be 1.0"
    assert math.isclose(metrics.f1, 0.6666, rel_tol=1e-04), "f1 should be 0.6666"
    assert math.isclose(metrics.accuracy, 0.6666, rel_tol=1e-04), "accuracy should be 0.6666"

#@pytest.fixture
def test_train_model():
    # Arrange
    data_df = pd.DataFrame({
        "id": list(range(100)),
        "numerical_feat": [random() * 100 for i in range(100)],
    })
    data_df["target_feat"] = [1 if i > 50 else 0 for i in data_df["numerical_feat"].values]

    model0 = LogisticRegression()
    model0_param_grid = {
        'C': [0.1, 1, 10, 100],
        'max_iter': [100, 200]}
    cv = 5
    X_train = data_df[['numerical_feat']]
    y_train = data_df['target_feat']

    # Act
    trained_model = model.train_model(X_train, y_train, model0, model0_param_grid, cv)
    params = trained_model.get_params()

    #Asserts
    assert isinstance(trained_model, LogisticRegression), "trained_model should be LogisticRegression"
    assert params['C'] in model0_param_grid['C'], f"C-param not in {model0_param_grid['C']}"
    assert params['max_iter'] in model0_param_grid['max_iter'], f"C-max_iter not in {model0_param_grid['max_iter']}"

#@pytest.fixture
def test_inference():
    # Arrange
    data_df = pd.DataFrame({
        "id": list(range(100)),
        "numerical_feat": [random() * 100 for i in range(100)],
    })
    data_df["target_feat"] = [1 if i > 50 else 0 for i in data_df["numerical_feat"].values]

    model0 = LogisticRegression()
    model0_param_grid = {
        'C': [0.1, 1],
        'max_iter': [100]}
    cv = 5
    X_train = data_df[['numerical_feat']]
    y_train = data_df['target_feat']

    # Act
    trained_model = model.train_model(X_train, y_train, model0, model0_param_grid, cv)
    preds = model.inference(trained_model, X_train)

    #Asserts
    assert hasattr(trained_model, 'predict'), "model does not have a 'predict' method"
    assert isinstance(preds, np.ndarray), "output should be of type np.ndarray"
    assert len(preds) == X_train.shape[0], "output and inputs should have the same number of rows"