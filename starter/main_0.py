# https://github.com/ympaik87/heroku_fastapi_deployment
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.train_model import CAT_FEATURES
from starter.ml.model import inference
from starter.ml.data import process_data


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()
MODEL_CONFIGS = {}


class InferenceRequest(BaseModel):
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
async def welcome():
    return "Welcome, this API returns predictions on Salary"


@app.post("/items")
async def create_item(item: dict):
    return item


@app.on_event("startup")
async def startup_event():
    cwd_p = os.getcwd()
    MODEL_CONFIGS['trained_model'] = \
        joblib.load(f"{cwd_p}/starter/model/model_trained.joblib")
    MODEL_CONFIGS['encoder'] = \
        joblib.load(f"{cwd_p}/starter/model/encoder.joblib")
    MODEL_CONFIGS['labels'] = \
        joblib.load(f"{cwd_p}/starter/model/lb.joblib")


@app.post('/predict')
async def get_prediction(request_data: InferenceRequest):
    request_dict = request_data.dict(by_alias=True)
    request_df = pd.DataFrame(request_dict, index=[0])
    processed_data, _, _, _ = process_data(
        request_df, categorical_features=CAT_FEATURES, label=None,
        training=False, encoder=MODEL_CONFIGS['encoder'],
        lb=MODEL_CONFIGS['labels']
    )
    preds = inference(MODEL_CONFIGS['trained_model'],
                      np.array(processed_data))
    if preds[0]:
        pred_cat = '>50K'
    else:
        pred_cat = '<=50K'

    return {"Predicted salary": pred_cat}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)