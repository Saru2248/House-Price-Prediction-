# -*- coding: utf-8 -*-
"""
predict.py
----------
Interactive CLI that takes property details from the user,
runs them through the saved pipeline, and returns the
predicted house price.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (
    engineer_features, encode_categoricals,
    scale_features, FEATURE_COLS,
)

# ── Valid choices shown to user ──────────────────────────────────────────────
VALID_LOCATIONS  = [
    "Bandra", "Andheri", "Thane", "Pune", "Navi Mumbai",
    "Borivali", "Dadar", "Worli", "Powai", "Nashik",
]
VALID_FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]


def _ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  ⚠️  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  ⚠️  Please enter an integer.")


def _ask_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return round(val, 1)
            print(f"  ⚠️  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  ⚠️  Please enter a number.")


def _ask_choice(prompt: str, choices: list) -> str:
    for i, c in enumerate(choices, 1):
        print(f"    {i}. {c}")
    while True:
        try:
            idx = int(input(prompt)) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"  ⚠️  Enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("  ⚠️  Please enter a number.")


def get_user_input() -> dict:
    """Collect property details interactively from the user."""
    print("\n" + "=" * 55)
    print("  🏠  HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 55)
    print("  Enter property details to predict the price.\n")

    area      = _ask_int ("  Area (sq.ft) [400–4000]         : ", 400, 4000)
    bedrooms  = _ask_int ("  Bedrooms      [1–5]             : ", 1, 5)
    bathrooms = _ask_int ("  Bathrooms     [1–5]             : ", 1, 5)
    balconies = _ask_int ("  Balconies     [0–3]             : ", 0, 3)

    print("\n  Select Location:")
    location  = _ask_choice("  Enter number  : ", VALID_LOCATIONS)

    age       = _ask_int ("  Age of property (years) [0–30]  : ", 0, 30)
    parking   = _ask_int ("  Parking spots  [0–2]            : ", 0, 2)

    print("\n  Select Furnishing:")
    furnishing = _ask_choice("  Enter number  : ", VALID_FURNISHING)

    floor        = _ask_int ("  Floor number   [0–30]           : ", 0, 30)
    total_floors = _ask_int ("  Total floors   [2–30]           : ", 2, 30)
    distance     = _ask_float("  Distance from city centre (km) : ", 0.5, 40)

    return {
        "area_sqft":        area,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "balconies":        balconies,
        "location":         location,
        "age_years":        age,
        "parking":          parking,
        "furnishing":       furnishing,
        "floor":            floor,
        "total_floors":     total_floors,
        "distance_city_km": distance,
        # dummy target so engineer_features doesn't break
        "price_lakhs":      0,
    }


def predict(input_dict: dict | None = None) -> float:
    """
    Predict house price.

    Parameters
    ----------
    input_dict : Property feature dict. If None, prompts user interactively.

    Returns
    -------
    Predicted price in Lakhs (float)
    """
    # Load artifacts
    model    = joblib.load("models/best_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    scaler   = joblib.load("models/scaler.pkl")

    if input_dict is None:
        input_dict = get_user_input()

    # Build DataFrame
    row = pd.DataFrame([input_dict])

    # Feature engineering (drop EDA-only column)
    row = engineer_features(row).drop(
        columns=["price_per_sqft", "price_lakhs"], errors="ignore"
    )

    # Encode & scale
    row, _ = encode_categoricals(row, encoders=encoders, fit=False)
    row, _ = scale_features(row, scaler=scaler, fit=False)

    # Align columns
    for col in FEATURE_COLS:
        if col not in row.columns:
            row[col] = 0
    row = row[FEATURE_COLS]

    predicted = model.predict(row)[0]
    predicted = round(float(predicted), 2)

    print("\n" + "=" * 55)
    print(f"  🏡  Predicted Price  : ₹ {predicted:.2f} Lakhs")
    if predicted >= 100:
        crore = predicted / 100
        print(f"                      = ₹ {crore:.2f} Crore")
    print("=" * 55)
    return predicted


if __name__ == "__main__":
    predict()
