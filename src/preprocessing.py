"""
preprocessing.py
----------------
Handles all data-cleaning, encoding, scaling, and
feature-engineering steps for the housing dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── 1. Data Cleaning ─────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with sensible defaults and remove
    extreme outliers from the price column.
    """
    df = df.copy()

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Remove price outliers using IQR method
    Q1 = df["price_lakhs"].quantile(0.25)
    Q3 = df["price_lakhs"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df["price_lakhs"] >= lower) & (df["price_lakhs"] <= upper)]
    print(f"  Outlier removal: {before} → {len(df)} rows "
          f"(removed {before - len(df)})")

    return df.reset_index(drop=True)


# ── 2. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional meaningful features from existing columns.
    """
    df = df.copy()

    # Price per sqft (derived later from model, so useful for EDA only)
    df["price_per_sqft"] = (df["price_lakhs"] * 1_00_000
                            / df["area_sqft"]).round(2)

    # Total rooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    # Floor ratio (position in building)
    df["floor_ratio"] = (
        df["floor"] / df["total_floors"].replace(0, 1)
    ).round(3)

    # Property newness score  (1 = brand new, 0 = 30 yrs old)
    df["newness_score"] = (1 - df["age_years"] / 30).clip(0, 1).round(3)

    # Amenity score
    df["amenity_score"] = (
        df["parking"] * 2 + df["balconies"] * 1
    )

    return df


# ── 3. Encoding ──────────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df       : Input DataFrame
    encoders : Pre-fitted encoders dict (for transform-only mode)
    fit      : True during training; False during inference

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    cat_cols = ["location", "furnishing"]

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


# ── 4. Scaling ───────────────────────────────────────────────────────────────

SCALE_COLS = [
    "area_sqft", "age_years", "distance_city_km",
    "floor_ratio", "newness_score", "amenity_score", "total_rooms",
]


def scale_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise numerical features.

    Parameters
    ----------
    df     : Encoded DataFrame
    scaler : Pre-fitted scaler (for transform-only mode)
    fit    : True during training; False during inference

    Returns
    -------
    (scaled_df, scaler)
    """
    df = df.copy()
    cols_present = [c for c in SCALE_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        df[cols_present] = scaler.transform(df[cols_present])

    return df, scaler


# ── 5. Full Pipeline ─────────────────────────────────────────────────────────

#: Columns actually used as model features (excludes EDA-only cols)
FEATURE_COLS = [
    "area_sqft", "bedrooms", "bathrooms", "balconies",
    "location", "age_years", "parking", "furnishing",
    "floor_ratio", "distance_city_km", "newness_score",
    "amenity_score", "total_rooms",
]

TARGET_COL = "price_lakhs"


def full_preprocess(
    df: pd.DataFrame,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict, StandardScaler]:
    """
    End-to-end preprocessing: clean → engineer → encode → scale.

    Returns
    -------
    X, y, encoders, scaler
    """
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders, fit=fit)
    df, scaler   = scale_features(df, scaler, fit=fit)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y, encoders, scaler
