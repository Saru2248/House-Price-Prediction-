# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Loads the saved best model and produces detailed evaluation
metrics + cross-validation scores on the full dataset.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import full_preprocess


def evaluate_model():
    """Load the best model and print comprehensive evaluation metrics."""

    # ── Load artifacts ───────────────────────────────────────────────────────
    best_model = joblib.load("models/best_model.pkl")
    encoders   = joblib.load("models/encoders.pkl")
    scaler     = joblib.load("models/scaler.pkl")

    df   = pd.read_csv("data/housing_data.csv")
    X, y, _, _ = full_preprocess(df, encoders=encoders,
                                  scaler=scaler, fit=False)

    y_pred = best_model.predict(X)

    # ── Metrics ──────────────────────────────────────────────────────────────
    mae   = mean_absolute_error(y, y_pred)
    rmse  = np.sqrt(mean_squared_error(y, y_pred))
    r2    = r2_score(y, y_pred)
    mape  = mean_absolute_percentage_error(y, y_pred) * 100

    print("\n" + "=" * 55)
    print("  FULL DATASET EVALUATION REPORT")
    print("=" * 55)
    print(f"  MAE   (Mean Absolute Error)       : ₹{mae:.2f} Lakhs")
    print(f"  RMSE  (Root Mean Squared Error)   : ₹{rmse:.2f} Lakhs")
    print(f"  R²    (Coefficient of Det.)       : {r2:.4f}")
    print(f"  MAPE  (Mean Abs. % Error)         : {mape:.2f}%")
    print("=" * 55)

    # ── 5-fold cross-validation ──────────────────────────────────────────────
    cv_scores = cross_val_score(best_model, X, y,
                                 cv=5, scoring="r2", n_jobs=-1)
    print(f"\n  5-Fold CV R² Scores : {cv_scores.round(4)}")
    print(f"  CV Mean R²          : {cv_scores.mean():.4f}")
    print(f"  CV Std Dev          : {cv_scores.std():.4f}")
    print("=" * 55)

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "R2": round(r2, 4),   "MAPE_pct": round(mape, 4),
        "CV_mean_R2": round(cv_scores.mean(), 4),
        "CV_std_R2":  round(cv_scores.std(), 4),
    }
    pd.DataFrame([report]).to_csv(
        "outputs/evaluation_report.csv", index=False
    )
    print("\n  Report saved → outputs/evaluation_report.csv")
    return report


if __name__ == "__main__":
    evaluate_model()
