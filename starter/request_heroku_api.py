import requests


req_data = {
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
req = requests.post('https://mlop3-fastapi-app.herokuapp.com/', json=req_data)

assert req.status_code == 200

print("Response code: ", req.status_code)
print("Response body: ", req.json())