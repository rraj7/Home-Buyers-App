from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow import keras

app = FastAPI()

# ------------------------------
# Input Schema
# ------------------------------
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# ------------------------------
# Helper for prediction
# ------------------------------
def make_prediction(model, features):
    return model.predict([features])[0]

# ------------------------------
# Load Models
# ------------------------------
linear_model = joblib.load("models/linear.pkl")
decision_tree_model = joblib.load("models/decision_tree.pkl")
random_forest_model = joblib.load("models/random_forest.pkl")
gradient_boosting_model = joblib.load("models/gradient_boosting.pkl")
neural_network_model = keras.models.load_model("models/neural_network.keras")

# ------------------------------
# Endpoints
# ------------------------------
@app.post("/predict/linear")
def predict_linear(features: HouseFeatures):
    vals = list(features.dict().values())
    pred = make_prediction(linear_model, vals)
    return {"model": "Linear Regression", "predicted_price": float(pred)}

@app.post("/predict/decision_tree")
def predict_dt(features: HouseFeatures):
    vals = list(features.dict().values())
    pred = make_prediction(decision_tree_model, vals)
    return {"model": "Decision Tree", "predicted_price": float(pred)}

@app.post("/predict/random_forest")
def predict_rf(features: HouseFeatures):
    vals = list(features.dict().values())
    pred = make_prediction(random_forest_model, vals)
    return {"model": "Random Forest", "predicted_price": float(pred)}

@app.post("/predict/gradient_boosting")
def predict_gb(features: HouseFeatures):
    vals = list(features.dict().values())
    pred = make_prediction(gradient_boosting_model, vals)
    return {"model": "Gradient Boosting", "predicted_price": float(pred)}

@app.post("/predict/neural")
def predict_neural(features: HouseFeatures):
    vals = np.array([list(features.dict().values())])
    pred = neural_network_model.predict(vals).flatten()[0]
    return {"model": "Neural Network", "predicted_price": float(pred)}
