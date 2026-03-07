from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
import time
import os

app = FastAPI(title="Flight Price Prediction API", description="Real-time inference API bridging ML prediction & UI")

# Load models at startup
model, scaler, encoders = None, None, None

@app.on_event("startup")
def load_assets():
    global model, scaler, encoders
    
    if not os.path.exists("models/rf_model.pkl"):
        # Just logging for now, it'll fail cleanly if the user hasn't trained it yet
        print("Warning: Model files not found. Please run train_model.py first.")
        return
        
    try:
        model = joblib.load("models/rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")
        print("ML Assets loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

class FlightInput(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    duration: float
    days_left: int
    stops: str
    flight_class: str
    
class BatchFlightInput(BaseModel):
    flights: List[FlightInput]

def preprocess_input(data: List[FlightInput]) -> pd.DataFrame:
    df = pd.DataFrame([vars(d) for d in data])
    # Mapping the pydantic model 'flight_class' back to the model's expected 'class'
    df = df.rename(columns={"flight_class": "class"})
    
    # Needs to match the columns used in train_model.py
    expected_cols = ["airline", "source_city", "destination_city", "departure_time", "arrival_time", "duration", "days_left", "stops", "class"]
    
    try:
        # Encode categorical variables
        cat_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city", "stops", "class"]
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])
            
        # Ensure order matches training
        df = df[expected_cols]
        
        # Scale
        scaled_features = scaler.transform(df)
        return scaled_features
    except Exception as e:
        # Re-raise standardizing exception
        raise ValueError(f"Preprocessing error. Check categorical values. Details: {e}")

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_single(data: FlightInput):
    """
    Predicts the price for a single flight. Latency tracked to ensure < 2ms objective.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
        
    start_time = time.time()
    try:
        features = preprocess_input([data])
        prediction = model.predict(features)[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    latency = time.time() - start_time
    
    return {
        "predicted_price": round(prediction, 2),
        "latency_seconds": round(latency, 4)
    }

@app.post("/simulate")
def predict_batch(data: BatchFlightInput):
    """
    Predicts prices for a batch of flights (50+ data points).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
        
    start_time = time.time()
    try:
        features = preprocess_input(data.flights)
        predictions = model.predict(features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    latency = time.time() - start_time
    
    return {
        "predicted_prices": [round(p, 2) for p in predictions],
        "total_scenarios": len(predictions),
        "latency_seconds": round(latency, 4)
    }
