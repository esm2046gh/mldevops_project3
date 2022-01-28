import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data
from starter.ml.data import Dump

import os
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()
trained_model = {}

class ModelInput(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Never-married")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Divorced")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

@app.get('/')
async def root():
    return "Udacity ML-DevOps Project3. Salary prediction."


@app.post("/items")
async def create_item(item: dict):
    return item

@app.on_event("startup")
async def on_startup():
    current_wdir = os.getcwd()
    print(f"CURRENT WORKING DIR: {current_wdir}")
    dump = Dump('joblib')
    trained_model['model'] = dump.load(f"{current_wdir}/starter/model/model.pkl")
    trained_model['encoder'] = dump.load(f"{current_wdir}/starter/model/encoder.pkl")
    trained_model['lb'] = dump.load(f"{current_wdir}/starter/model/lb.pkl")
    trained_model['output_feature'] = dump.load(f"{current_wdir}/starter/model/output_feature.pkl")
    trained_model['scaler'] = dump.load(f"{current_wdir}/starter/model/scaler.pkl")
    trained_model['cat_features'] = dump.load(f"{current_wdir}/starter/model/cat_features.pkl")

@app.post('/predict')
async def predict(input_data: ModelInput):
    input_dict = input_data.dict(by_alias=True)
    input_df = pd.DataFrame(input_dict, index=[0])
    X_data, _, _, _, _ = process_data(
    X=input_df,
    categorical_features=trained_model['cat_features'],
    label= None, #trained_model['output_feature'],
    training= False,
    encoder= trained_model['encoder'],
    lb= trained_model['lb'],
    scaler=trained_model['scaler'])

    preds = inference(trained_model['model'], X_data)
    preds_str = '>50K' if preds[0] else '<=50K'

    return {"Predicted salary": preds_str}


# if __name__ == "__main__":
#     import uvicorn
#     #uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
#     uvicorn.run("main:app", reload=True)
# CLI: $ uvicorn starter.main:app --reload