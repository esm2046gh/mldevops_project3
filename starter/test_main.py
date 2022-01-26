from fastapi.testclient import TestClient
import pytest
from main import app

@pytest.fixture
def client():
    with TestClient(app) as cli:
        yield cli

def test_get_root(client):
    ret = client.get('/')
    assert ret.status_code == 200, "status_code should be 200"
    assert ret.json() == "Udacity ML-DevOps Project3. Salary prediction.", "Wrong json output"

def test_post_items(client):
    sample_dict = {
        "age": 49,
        "workclass": "State-gov",
    }
    ret = client.post('/items', json=sample_dict)
    assert ret.status_code == 200, "status_code should be 200"
    assert ret.json() == sample_dict


def test_post_predict_leq_50k(client):
    input_dict = {
        "age": 49,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    ret = client.post('/predict', json=input_dict)
    assert ret.status_code == 200, "status_code should be 200"
    assert ret.json() == {"Predicted salary": "<=50K"}, "Wrong json output"


def test_post_predict_bt_50k(client):
    input_dict = {
        "age": 41,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2020,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    ret = client.post("/predict", json=input_dict)
    assert ret.status_code == 200, "status_code should be 200"
    assert ret.json() == {"Predicted salary": ">50K"}, "Wrong json output"
