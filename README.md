# SkyPrice ✈️ — Smart Flight Fare Intelligence

A production-ready ML system that predicts Indian domestic flight fares, compares airlines, and identifies the cheapest day to fly. Built with a **FastAPI** inference backend and a **Streamlit** analytics frontend.

---

## 📁 Project Structure--

```
flight_project/
├── assets/
│   └── hero.png               # UI hero banner image
├── models/
│   ├── rf_model.pkl           # Trained RandomForest model
│   ├── scaler.pkl             # StandardScaler
│   ├── encoders.pkl           # LabelEncoders per categorical column
│   └── model_meta.json        # Training metadata (R², MAE, timestamp)
├── .streamlit/
│   └── config.toml            # Streamlit theme config
├── api.py                     # FastAPI inference server
├── app.py                     # Streamlit frontend
├── train_model.py             # Model training pipeline
├── api_test.py                # pytest test suite
├── requirements.txt           # Pinned dependencies
└── airlines_flights_data.csv  # Raw dataset
```

---

## ⚙️ Setup

### 1. Create & activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This will produce `models/rf_model.pkl`, `models/scaler.pkl`, `models/encoders.pkl`, and `models/model_meta.json`.

---

## 🚀 Running the App

You need **two terminals** running simultaneously.

### Terminal 1 — Start the FastAPI inference server
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2 — Start the Streamlit UI
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🔌 API Reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Rich health check — model status, metadata |
| `GET` | `/model-info` | Training metrics (R², MAE, timestamp) |
| `GET` | `/feature-importance` | Feature importance scores from the RF model |
| `POST` | `/predict` | Single flight price prediction |
| `POST` | `/predict-range` | Prediction with confidence interval (±range) |
| `POST` | `/simulate` | Batch prediction for up to 60 scenarios |

### POST `/predict` — Request Body
```json
{
  "airline": "Vistara",
  "source_city": "Delhi",
  "destination_city": "Mumbai",
  "departure_time": "Morning",
  "arrival_time": "Afternoon",
  "duration": 2.5,
  "days_left": 15,
  "stops": "zero",
  "flight_class": "Economy"
}
```

### Valid Values

| Field | Valid Options |
|---|---|
| `airline` | `SpiceJet`, `AirAsia`, `Vistara`, `GO_FIRST`, `Indigo`, `Air_India` |
| `source_city` / `destination_city` | `Delhi`, `Mumbai`, `Bangalore`, `Kolkata`, `Hyderabad`, `Chennai` |
| `departure_time` / `arrival_time` | `Early_Morning`, `Morning`, `Afternoon`, `Evening`, `Night`, `Late_Night` |
| `stops` | `zero`, `one`, `two_or_more` |
| `flight_class` | `Economy`, `Business` |
| `days_left` | Integer `1–365` |
| `duration` | Float `0.5–20.0` hours |

### Rate Limits
- `/predict` — 60 requests / minute
- `/simulate` — 10 requests / minute

---

## 🧪 Running Tests
```bash
pytest api_test.py -v
```

Tests cover: happy-path predictions, batch simulation, invalid input (422 validation), health checks, and latency assertions.

---

## 🗂️ Dataset

`airlines_flights_data.csv` — Indian domestic flight data with columns:
`airline`, `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`, `duration`, `days_left`, `price`

---

## 📊 Model Performance

| Metric | Baseline (Linear Reg.) | Optimized (Random Forest) |
|---|---|---|
| R² Score | ~0.91 | ~0.98 |
| MAE | ~Rs. 3,800 | ~Rs. 1,100 |

Exact values are written to `models/model_meta.json` after training.
