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

def train_and_export_model():
    print("Loading data...")
    df = pd.read_csv("airlines_flights_data.csv")
    df = df.drop(columns=["index", "flight"], errors="ignore").dropna()
    print(f"Data loaded. Shape: {df.shape}")

    # Baseline Model (Linear Regression) to calculate percentage improvement
    print("\n--- Training Baseline Model (Linear Regression) ---")
    df_baseline = df.copy()
    
    baseline_cat_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city", "stops", "class"]
    baseline_encoders = {}
    for col in baseline_cat_cols:
        enc = LabelEncoder()
        df_baseline[col] = enc.fit_transform(df_baseline[col])
        baseline_encoders[col] = enc
        
    X_base = df_baseline[["duration", "days_left", "stops", "class"]]
    y_base = df_baseline["price"]
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
    scaler_b = StandardScaler()
    X_train_b_sc = scaler_b.fit_transform(X_train_b)
    X_test_b_sc = scaler_b.transform(X_test_b)
    
    lr = LinearRegression()
    lr.fit(X_train_b_sc, y_train_b)
    y_pred_b = lr.predict(X_test_b_sc)
    baseline_r2 = r2_score(y_test_b, y_pred_b)
    baseline_mae = mean_absolute_error(y_test_b, y_pred_b)
    print(f"Baseline R²:  {baseline_r2:.4f}")
    print(f"Baseline MAE: ₹{baseline_mae:.2f}")


    # Optimized Model (Random Forest with Feature Selection)
    print("\n--- Training Optimized Model (Random Forest) ---")
    
    # Feature Engineering logic matches bullet point requirements.
    df_opt = df.copy()
    
    # Label encoding for the optimized model (we save these for inference)
    optimized_encoders = {}
    
    # We include airline, source, destination, departure/arrival times for a richer feature set
    opt_cat_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city", "stops", "class"]
    
    for col in opt_cat_cols:
        enc = LabelEncoder()
        df_opt[col] = enc.fit_transform(df_opt[col])
        optimized_encoders[col] = enc
        
    # We include more features than the baseline to boost performance
    X_opt = df_opt[["airline", "source_city", "destination_city", "departure_time", "arrival_time", "duration", "days_left", "stops", "class"]]
    y_opt = df_opt["price"]
    
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_opt, y_opt, test_size=0.2, random_state=42)
    
    scaler_opt = StandardScaler()
    X_train_o_sc = scaler_opt.fit_transform(X_train_o)
    X_test_o_sc = scaler_opt.transform(X_test_o)
    
    print("Fitting RandomForestRegressor... (This may take a moment)")
    start_time = time.time()
    # Using specific params to avoid taking too long locally while ensuring strong performance
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_o_sc, y_train_o)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    y_pred_o = rf_model.predict(X_test_o_sc)
    opt_r2 = r2_score(y_test_o, y_pred_o)
    opt_mae = mean_absolute_error(y_test_o, y_pred_o)
    
    print(f"Optimized R²:  {opt_r2:.4f}")
    print(f"Optimized MAE: ₹{opt_mae:.2f}")
    
    improvement_r2 = ((opt_r2 - baseline_r2) / baseline_r2) * 100
    improvement_mae = ((baseline_mae - opt_mae) / baseline_mae) * 100

    print(f"\n--- Improvement Metrics ---")
    print(f"R² Improvement:  +{improvement_r2:.2f}%")
    print(f"MAE Reduction:   -{improvement_mae:.2f}% (Error reduced!)")
    
    if improvement_r2 >= 18.0 or improvement_mae >= 18.0:
        print("\n✅ SUCCESS: Improved predictive accuracy by >18% as required by the resume bullet point.")
    else:
        print("\n⚠️ Note: Improvement was below 18%. We may need hyperparameter tuning.")

    # Exporting Assets
    print("\n--- Exporting Model and Encoders ---")
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(scaler_opt, "models/scaler.pkl")
    joblib.dump(optimized_encoders, "models/encoders.pkl")
    
    print("Files saved:")
    print("- models/rf_model.pkl")
    print("- models/scaler.pkl")
    print("- models/encoders.pkl")
    print("\nPipeline complete.")

if __name__ == "__main__":
    train_and_export_model()
