import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import time
import json
import logging
from datetime import datetime, timezone

# ── Logging setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def train_and_export_model():
    # ── Load data ────────────────────────────────────────────────────────
    log.info("Loading dataset...")
    df = pd.read_csv("airlines_flights_data.csv")
    df = df.drop(columns=["index", "flight"], errors="ignore").dropna()
    log.info("Dataset loaded — shape: %s", df.shape)

    cat_cols = [
        "airline", "source_city", "departure_time",
        "arrival_time", "destination_city", "stops", "class",
    ]

    # ── Baseline model (Linear Regression) ───────────────────────────────
    log.info("Training baseline model (Linear Regression)...")
    df_baseline = df.copy()
    baseline_encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        df_baseline[col] = enc.fit_transform(df_baseline[col])
        baseline_encoders[col] = enc

    X_base = df_baseline[["duration", "days_left", "stops", "class"]]
    y_base = df_baseline["price"]
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42
    )
    scaler_b = StandardScaler()
    lr = LinearRegression()
    lr.fit(scaler_b.fit_transform(X_tr_b), y_tr_b)
    y_pred_b = lr.predict(scaler_b.transform(X_te_b))
    baseline_r2 = r2_score(y_te_b, y_pred_b)
    baseline_mae = mean_absolute_error(y_te_b, y_pred_b)
    log.info("Baseline  R²=%.4f  MAE=Rs.%.2f", baseline_r2, baseline_mae)

    # ── Optimised model (Random Forest) ──────────────────────────────────
    log.info("Training optimised model (Random Forest)...")
    df_opt = df.copy()
    optimised_encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        df_opt[col] = enc.fit_transform(df_opt[col])
        optimised_encoders[col] = enc

    feature_cols = [
        "airline", "source_city", "destination_city",
        "departure_time", "arrival_time",
        "duration", "days_left", "stops", "class",
    ]
    X_opt = df_opt[feature_cols]
    y_opt = df_opt["price"]
    X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
        X_opt, y_opt, test_size=0.2, random_state=42
    )
    scaler_opt = StandardScaler()
    X_tr_o_sc = scaler_opt.fit_transform(X_tr_o)
    X_te_o_sc = scaler_opt.transform(X_te_o)

    rf_params = dict(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_model = RandomForestRegressor(**rf_params)
    t0 = time.time()
    rf_model.fit(X_tr_o_sc, y_tr_o)
    train_time = time.time() - t0
    log.info("Training complete in %.1fs", train_time)

    y_pred_o = rf_model.predict(X_te_o_sc)
    opt_r2 = r2_score(y_te_o, y_pred_o)
    opt_mae = mean_absolute_error(y_te_o, y_pred_o)
    improvement_r2 = ((opt_r2 - baseline_r2) / baseline_r2) * 100
    improvement_mae = ((baseline_mae - opt_mae) / baseline_mae) * 100

    log.info("Optimised R²=%.4f  MAE=Rs.%.2f", opt_r2, opt_mae)
    log.info(
        "Improvement: R² +%.2f%%  |  MAE reduced by %.2f%%",
        improvement_r2,
        improvement_mae,
    )

    if improvement_r2 >= 18.0 or improvement_mae >= 18.0:
        log.info("SUCCESS — improved predictive accuracy by >18%% (resume target met).")
    else:
        log.warning(
            "Improvement below 18%%. Consider hyperparameter tuning."
        )

    # ── Export model artifacts ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model,           "models/rf_model.pkl")
    joblib.dump(scaler_opt,         "models/scaler.pkl")
    joblib.dump(optimised_encoders, "models/encoders.pkl")
    log.info("Model artifacts saved to models/")

    # ── Export metadata JSON ──────────────────────────────────────────────
    feature_importances = dict(
        zip(feature_cols, rf_model.feature_importances_.tolist())
    )
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(X_tr_o)),
        "test_samples": int(len(X_te_o)),
        "model": "RandomForestRegressor",
        "hyperparameters": rf_params,
        "training_time_seconds": round(train_time, 2),
        "metrics": {
            "r2_score": round(opt_r2, 6),
            "mae": round(opt_mae, 2),
            "baseline_r2": round(baseline_r2, 6),
            "baseline_mae": round(baseline_mae, 2),
            "improvement_r2_pct": round(improvement_r2, 2),
            "improvement_mae_pct": round(improvement_mae, 2),
        },
        "feature_importance": {
            k: round(v, 6) for k, v in
            sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        },
        "features": feature_cols,
    }
    meta_path = "models/model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written to %s", meta_path)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    train_and_export_model()
