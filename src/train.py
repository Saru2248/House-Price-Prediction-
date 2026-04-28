# -*- coding: utf-8 -*-
"""
train.py
--------
Trains three regression models, evaluates them, saves the best
model + preprocessing artifacts, and prints a comparison table.

Models trained:
    1. Linear Regression
    2. Random Forest Regressor
    3. XGBoost Regressor
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor

# ── ensure project root is importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator  import generate_housing_data
from src.preprocessing   import full_preprocess

MODELS_DIR = "models"
DATA_DIR   = "data"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


# ── 1. Metric helper ────────────────────────────────────────────────────────

def evaluate(name: str, model, X_test, y_test) -> dict:
    """Compute MAE, RMSE, R² and return as dict."""
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Model  : {name}")
    print(f"  MAE    : ₹{mae:.2f} Lakhs")
    print(f"  RMSE   : ₹{rmse:.2f} Lakhs")
    print(f"  R²     : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"{'='*50}")
    return {"Model": name, "MAE": round(mae, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ── 2. Main training function ────────────────────────────────────────────────

def train():
    print("\n📦 Generating / loading dataset ...")
    csv_path = os.path.join(DATA_DIR, "housing_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded from {csv_path}  |  shape: {df.shape}")
    else:
        df = generate_housing_data(2000)
        df.to_csv(csv_path, index=False)
        print(f"  Generated & saved → {csv_path}  |  shape: {df.shape}")

    print("\n⚙️  Preprocessing ...")
    X, y, encoders, scaler = full_preprocess(df, fit=True)
    print(f"  Feature matrix  : {X.shape}")
    print(f"  Target (sample) : {y.head(5).tolist()}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # ── 3. Define models ─────────────────────────────────────────────────────
    models = {
        "Linear Regression":      LinearRegression(),
        "Decision Tree":          DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":          RandomForestRegressor(
                                      n_estimators=200, max_depth=12,
                                      random_state=42, n_jobs=-1),
        "XGBoost":                XGBRegressor(
                                      n_estimators=300, learning_rate=0.05,
                                      max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, random_state=42,
                                      verbosity=0),
    }

    # ── 4. Train & evaluate ──────────────────────────────────────────────────
    results  = []
    trained  = {}
    print("\n🏋️  Training models ...")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  ✅ {name} trained in {elapsed:.1f}s")
        metrics = evaluate(name, model, X_test, y_test)
        metrics["Train_sec"] = round(elapsed, 2)
        results.append(metrics)
        trained[name] = model

    # ── 5. Comparison table ──────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    results_df.to_csv(os.path.join("outputs", "model_comparison.csv"), index=False)
    print("\n\n📊 MODEL COMPARISON TABLE")
    print(results_df.to_string(index=False))

    # ── 6. Save artifacts ────────────────────────────────────────────────────
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained[best_name]
    print(f"\n🏆 Best model: {best_name} (R² = {results_df.iloc[0]['R2']})")

    joblib.dump(best_model,  os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(encoders,    os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(trained,     os.path.join(MODELS_DIR, "all_models.pkl"))
    print(f"  Artifacts saved to '{MODELS_DIR}/' folder")

    # Save test set for visualization
    test_df             = X_test.copy()
    test_df["actual"]   = y_test.values
    test_df["predicted"] = best_model.predict(X_test)
    test_df.to_csv(os.path.join("outputs", "test_predictions.csv"), index=False)

    print("\n✅ Training complete!")
    return trained, encoders, scaler, results_df


if __name__ == "__main__":
    train()
