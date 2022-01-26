"""
- Module to test model.py
- This is part of Project-3 of Machine Learning DevOps Nano Degree

Author: E. Saavedra
Date: Jan. 2022
"""
import sys
import logging
from dataclasses import dataclass

def press_c_to_continue():
    """
    - Controls the workflow of this script prompting for 'c'-> continue or 'x'->exit
    """
    while True:
        kb_pressed = input("Press (c)ontinue, e(x)it > ")
        if kb_pressed == 'c':
            break
        if kb_pressed == 'x':
            sys.exit("Script execution aborted")

def test_train_model(X_train, y_train):
    """
    - Tests train_model.
    """
    print('>>> Running ' + test_train_model.__name__)

    try:
        assert X_train.shape[0] > 0
        assert len(y_train.shape) == X_train.shape[0]
        logging.info("Testing import_data: SUCCESS")
        print('<<< Finished ' + test_train_model.__name__)
        # return objects for further usage in testing
    except AssertionError as err:
        print(f"{test_train_model.__name__}: Number of rows in Xtrain and y_train are not equal")
        raise err


def test_compute_model_metrics(y, preds):
    """
    - Tests compute_model_metrics
    """
    print('>>> Running ' + test_compute_model_metrics.__name__)
    try:
        assert y is not null
        assert preds is not null
    except AssertionError as err:
        print(f"{test_compute_model_metrics.__name__}: Failed")
    raise err

def test_inference(model, X):
    """
    - Tests models inference
    """
    print('>>> Running ' + test_inference.__name__)
    try:
        assert model is not null
        assert X is not null
    except AssertionError as err:
        print(f"{test_inference.__name__}: Failed")
    raise err


if __name__ == "__main__":
    print('\n=========================================================')
    print('=============== Program to test model.py ================\n')
    print('- The program will execute the following steps:')
    print('  1. test_train_model')
    print('  2. test_compute_model_metrics')
    print('  3. test_inference')
    press_c_to_continue()
    print(
        '>>> Running script: ' +
        __file__ )


    test_train_model(X_train, y_train)
    press_c_to_continue()
    test_compute_model_metrics(y, preds)
    press_c_to_continue()
    test_inference(model, X)
    print('>>> End script: ' + __file__)
    press_c_to_continue()