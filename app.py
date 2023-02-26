import models.ml.classifier as knn
from fastapi import FastAPI, Body
from joblib import load
from models.ml.iris import Iris

title = "Machine Learning Iris API"
description = "API Dataset ML Model"
version = "1.0"

app = FastAPI(title=title, description=description, version=version)

@app.on_event('startup')
async def load_model():
    knn.model = load('models/ml/iris_dt_v1.joblib')

@app.post('/predict', tags=["predictions"])
async def get_prediction(iris : Iris):
    data = dict(iris)['data']
    prediction = knn.model.predict(data).tolist()
    log_proba = knn.model.predict_proba(data).tolist()
    return {"prediction":prediction, "log_proba":log_proba}