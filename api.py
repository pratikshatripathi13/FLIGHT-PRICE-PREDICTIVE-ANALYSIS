from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import time
import os
import numpy as np

app = FastAPI(
    title="Flight Price Prediction API",
    description="Real-time ML inference API for predicting flight prices",
    version="2.0.0"
)

# -----------------------------
# Load ML Assets
# -----------------------------
model = None
scaler = None
encoders = None


@app.on_event("startup")
def load_assets():
    global model, scaler, encoders

    try:
        model = joblib.load("models/rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")

        print("✅ ML assets loaded successfully.")

    except FileNotFoundError:
        print("❌ Model files not found. Run train_model.py first.")

    except Exception as e:
        print(f"❌ Error loading ML assets: {e}")


# -----------------------------
# Input Schemas
# -----------------------------
class FlightInput(BaseModel):
    airline: str = Field(..., example="Indigo")
    source_city: str = Field(..., example="Delhi")
    destination_city: str = Field(..., example="Mumbai")
    departure_time: str = Field(..., example="Morning")
    arrival_time: str = Field(..., example="Night")
    duration: float = Field(..., gt=0, example=2.5)
    days_left: int = Field(..., ge=0, example=10)
    stops: str = Field(..., example="zero")
    flight_class: str = Field(..., example="Economy")


class BatchFlightInput(BaseModel):
    flights: List[FlightInput]


# -----------------------------
# Output Schemas
# -----------------------------
class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predicted_prices: List[float]
    total_scenarios: int
    latency_ms: float


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_input(data: List[FlightInput]):

    if encoders is None or scaler is None:
        raise ValueError("ML preprocessing assets are unavailable.")

    df = pd.DataFrame([item.dict() for item in data])

    # Rename column
    df = df.rename(columns={"flight_class": "class"})

    expected_cols = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "duration",
        "days_left",
        "stops",
        "class"
    ]

    categorical_cols = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "stops",
        "class"
    ]

    try:
        # Validate unseen categories
        for col in categorical_cols:

            unknown_values = set(df[col]) - set(encoders[col].classes_)

            if unknown_values:
                raise ValueError(
                    f"Invalid value(s) in '{col}': {list(unknown_values)}"
                )

            df[col] = encoders[col].transform(df[col])

        # Reorder columns
        df = df[expected_cols]

        # Scale features
        scaled_features = scaler.transform(df)

        return scaled_features

    except Exception as e:
        raise ValueError(str(e))


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Flight Price Prediction API is running",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "model_type": type(model).__name__,
        "features_expected": 9
    }


# -----------------------------
# Single Prediction
# -----------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_single(data: FlightInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first."
        )

    start = time.perf_counter()

    try:
        features = preprocess_input([data])

        prediction = model.predict(features)[0]

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR",
        "latency_ms": round(latency_ms, 2)
    }


# -----------------------------
# Batch Prediction
# -----------------------------
@app.post(
    "/simulate",
    response_model=BatchPredictionResponse
)
def predict_batch(data: BatchFlightInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first."
        )

    start = time.perf_counter()

    try:
        features = preprocess_input(data.flights)

        predictions = model.predict(features)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "predicted_prices": [
            round(float(p), 2) for p in predictions
        ],
        "total_scenarios": len(predictions),
        "latency_ms": round(latency_ms, 2)
    }