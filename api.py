"""
SkyPrice — Flight Price Prediction API
Production-ready FastAPI inference server.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("skyprice.api")

# ── Valid categorical values (single source of truth) ────────────────────
VALID_AIRLINES     = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]
VALID_CITIES       = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
VALID_TIMES        = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
VALID_STOPS        = ["zero", "one", "two_or_more"]
VALID_CLASSES      = ["Economy", "Business"]
FEATURE_COLS       = [
    "airline", "source_city", "destination_city",
    "departure_time", "arrival_time",
    "duration", "days_left", "stops", "class",
]

# ── Global model state ────────────────────────────────────────────────────
_state: dict = {"model": None, "scaler": None, "encoders": None, "meta": None, "loaded_at": None}


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML assets on startup; log clean shutdown."""
    model_path = "models/rf_model.pkl"
    if not os.path.exists(model_path):
        log.warning("Model files not found. Run train_model.py first.")
    else:
        try:
            t0 = time.time()
            _state["model"]    = joblib.load("models/rf_model.pkl")
            _state["scaler"]   = joblib.load("models/scaler.pkl")
            _state["encoders"] = joblib.load("models/encoders.pkl")
            _state["loaded_at"] = time.time()
            load_ms = (time.time() - t0) * 1000

            meta_path = "models/model_meta.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    _state["meta"] = json.load(f)

            log.info("ML assets loaded in %.1f ms.", load_ms)
        except Exception as exc:
            log.error("Failed to load model assets: %s", exc)

    yield  # — application runs here —

    log.info("SkyPrice API shutting down.")


# ── Rate limiter ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ── App factory ───────────────────────────────────────────────────────────
app = FastAPI(
    title="SkyPrice — Flight Price Prediction API",
    description=(
        "Production-ready ML inference API for Indian domestic flight fare prediction. "
        "Supports single predictions, confidence intervals, batch simulation, and model introspection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────
class FlightInput(BaseModel):
    airline:          Literal["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]
    source_city:      Literal["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    destination_city: Literal["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    departure_time:   Literal["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    arrival_time:     Literal["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    duration:         float = Field(..., gt=0, le=20, description="Flight duration in hours (0–20)")
    days_left:        int   = Field(..., ge=1, le=365, description="Days until departure (1–365)")
    stops:            Literal["zero", "one", "two_or_more"]
    flight_class:     Literal["Economy", "Business"]

    model_config = {"json_schema_extra": {
        "example": {
            "airline": "Vistara", "source_city": "Delhi", "destination_city": "Mumbai",
            "departure_time": "Morning", "arrival_time": "Afternoon",
            "duration": 2.5, "days_left": 15, "stops": "zero", "flight_class": "Economy",
        }
    }}


class BatchFlightInput(BaseModel):
    flights: List[FlightInput] = Field(..., min_length=1, max_length=200)


# ── Preprocessing helper ──────────────────────────────────────────────────
def _preprocess(flights: List[FlightInput]) -> np.ndarray:
    """Encode and scale a list of FlightInput objects → numpy array."""
    df = pd.DataFrame([f.model_dump() for f in flights])
    df = df.rename(columns={"flight_class": "class"})
    enc = _state["encoders"]
    cat_cols = ["airline", "source_city", "departure_time",
                "arrival_time", "destination_city", "stops", "class"]
    try:
        for col in cat_cols:
            df[col] = enc[col].transform(df[col])
        df = df[FEATURE_COLS]
        return _state["scaler"].transform(df)
    except Exception as exc:
        raise ValueError(f"Preprocessing failed: {exc}") from exc


def _require_model():
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run train_model.py first, then restart the API.",
        )


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health(request: Request):
    """Rich health check — returns model status, uptime, and training metadata."""
    model_loaded = _state["model"] is not None
    uptime = round(time.time() - _state["loaded_at"], 1) if _state["loaded_at"] else None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "uptime_seconds": uptime,
        "meta": _state["meta"],
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info(request: Request):
    """Returns training metadata: R², MAE, training timestamp, hyperparameters."""
    _require_model()
    if not _state["meta"]:
        raise HTTPException(
            status_code=404,
            detail="model_meta.json not found. Re-run train_model.py to generate it.",
        )
    return _state["meta"]


@app.get("/feature-importance", tags=["Monitoring"])
def feature_importance(request: Request):
    """Returns per-feature importance scores from the trained Random Forest."""
    _require_model()
    model = _state["model"]
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    return {"feature_importance": sorted_imp, "features": FEATURE_COLS}


@app.post("/predict", tags=["Inference"])
@limiter.limit("60/minute")
def predict_single(data: FlightInput, request: Request):
    """
    Predicts the price for a single flight.
    Rate limited to 60 requests/minute per IP.
    """
    _require_model()
    t0 = time.time()
    try:
        features = _preprocess([data])
        price = float(_state["model"].predict(features)[0])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    latency = time.time() - t0
    log.info("predict  price=%.0f  latency=%.4fs", price, latency)
    return {"predicted_price": round(price, 2), "latency_seconds": round(latency, 4)}


@app.post("/predict-range", tags=["Inference"])
@limiter.limit("60/minute")
def predict_range(data: FlightInput, request: Request):
    """
    Predicts price with a confidence interval derived from individual tree estimates.
    Returns p10, median, p90 alongside the mean prediction.
    """
    _require_model()
    t0 = time.time()
    try:
        features = _preprocess([data])
        tree_preds = np.array([
            tree.predict(features)[0]
            for tree in _state["model"].estimators_
        ])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    latency = time.time() - t0
    result = {
        "predicted_price": round(float(np.mean(tree_preds)), 2),
        "p10": round(float(np.percentile(tree_preds, 10)), 2),
        "p90": round(float(np.percentile(tree_preds, 90)), 2),
        "std_dev": round(float(np.std(tree_preds)), 2),
        "latency_seconds": round(latency, 4),
    }
    log.info(
        "predict-range  price=%.0f  p10=%.0f  p90=%.0f  latency=%.4fs",
        result["predicted_price"], result["p10"], result["p90"], latency,
    )
    return result


@app.post("/simulate", tags=["Inference"])
@limiter.limit("10/minute")
def predict_batch(data: BatchFlightInput, request: Request):
    """
    Predicts prices for a batch of up to 200 flight scenarios.
    Rate limited to 10 requests/minute per IP.
    """
    _require_model()
    t0 = time.time()
    try:
        features = _preprocess(data.flights)
        predictions = _state["model"].predict(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    latency = time.time() - t0
    log.info("simulate  n=%d  latency=%.4fs", len(predictions), latency)
    return {
        "predicted_prices": [round(float(p), 2) for p in predictions],
        "total_scenarios": len(predictions),
        "latency_seconds": round(latency, 4),
    }
