# -*- coding: utf-8 -*-
"""
app.py
------
FastAPI backend for the House Price Prediction web dashboard.

Endpoints:
  GET  /                     → Serve the main HTML dashboard
  GET  /api/stats            → Dataset & model statistics
  GET  /api/model-results    → Model comparison metrics
  GET  /api/charts/{name}    → Serve saved chart PNGs
  POST /api/predict          → Predict price for new property
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── lazy model loading (only when first request arrives) ─────────────────────
_model = _encoders = _scaler = None

def _load_artifacts():
    global _model, _encoders, _scaler
    if _model is None:
        import joblib
        _model    = joblib.load("models/best_model.pkl")
        _encoders = joblib.load("models/encoders.pkl")
        _scaler   = joblib.load("models/scaler.pkl")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="House Price Prediction", version="1.0")

# Static files & templates
app.mount("/static",  StaticFiles(directory="static"),    name="static")
app.mount("/images",  StaticFiles(directory="images"),    name="images")
templates = Jinja2Templates(directory="templates")


# ── Pydantic input schema ─────────────────────────────────────────────────────
class PropertyInput(BaseModel):
    area_sqft:        float
    bedrooms:         int
    bathrooms:        int
    balconies:        int
    location:         str
    age_years:        int
    parking:          int
    furnishing:       str
    floor:            int
    total_floors:     int
    distance_city_km: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats")
async def get_stats():
    """Return summary statistics from the dataset."""
    try:
        df = pd.read_csv("data/housing_data.csv")
        stats = {
            "total_records":   int(len(df)),
            "avg_price":       round(float(df["price_lakhs"].mean()), 2),
            "min_price":       round(float(df["price_lakhs"].min()), 2),
            "max_price":       round(float(df["price_lakhs"].max()), 2),
            "avg_area":        round(float(df["area_sqft"].mean()), 0),
            "locations":       int(df["location"].nunique()),
            "avg_age":         round(float(df["age_years"].mean()), 1),
            "price_distribution": {
                "labels": ["<20L", "20-50L", "50-100L", "100-200L", ">200L"],
                "values": [
                    int((df["price_lakhs"] < 20).sum()),
                    int(((df["price_lakhs"] >= 20) & (df["price_lakhs"] < 50)).sum()),
                    int(((df["price_lakhs"] >= 50) & (df["price_lakhs"] < 100)).sum()),
                    int(((df["price_lakhs"] >= 100) & (df["price_lakhs"] < 200)).sum()),
                    int((df["price_lakhs"] >= 200).sum()),
                ]
            },
            "location_avg_price": df.groupby("location")["price_lakhs"]
                                     .mean().round(2)
                                     .sort_values(ascending=False)
                                     .to_dict(),
            "bedroom_counts": df["bedrooms"].value_counts().sort_index().to_dict(),
            "furnishing_counts": df["furnishing"].value_counts().to_dict(),
        }
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-results")
async def get_model_results():
    """Return model comparison metrics."""
    try:
        comp_path = "outputs/model_comparison.csv"
        eval_path = "outputs/evaluation_report.csv"

        models_data = []
        if os.path.exists(comp_path):
            df = pd.read_csv(comp_path)
            for _, row in df.iterrows():
                models_data.append({
                    "name":   row["Model"],
                    "mae":    round(float(row["MAE"]),  4),
                    "rmse":   round(float(row["RMSE"]), 4),
                    "r2":     round(float(row["R2"]),   4),
                    "r2_pct": round(float(row["R2"]) * 100, 2),
                })

        eval_data = {}
        if os.path.exists(eval_path):
            edf = pd.read_csv(eval_path)
            eval_data = {
                "mae":        round(float(edf["MAE"].iloc[0]),  4),
                "rmse":       round(float(edf["RMSE"].iloc[0]), 4),
                "r2":         round(float(edf["R2"].iloc[0]),   4),
                "mape":       round(float(edf["MAPE_pct"].iloc[0]), 2),
                "cv_mean_r2": round(float(edf["CV_mean_R2"].iloc[0]), 4),
                "cv_std_r2":  round(float(edf["CV_std_R2"].iloc[0]),  4),
            }

        return JSONResponse({"models": models_data, "best_eval": eval_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict_price(data: PropertyInput):
    """Predict house price for given property features."""
    try:
        _load_artifacts()
        from src.preprocessing import engineer_features, encode_categoricals, \
                                      scale_features, FEATURE_COLS

        row = pd.DataFrame([{
            "area_sqft":        data.area_sqft,
            "bedrooms":         data.bedrooms,
            "bathrooms":        data.bathrooms,
            "balconies":        data.balconies,
            "location":         data.location,
            "age_years":        data.age_years,
            "parking":          data.parking,
            "furnishing":       data.furnishing,
            "floor":            data.floor,
            "total_floors":     data.total_floors,
            "distance_city_km": data.distance_city_km,
            "price_lakhs":      0,
        }])

        row = engineer_features(row).drop(
            columns=["price_per_sqft", "price_lakhs"], errors="ignore"
        )
        row, _ = encode_categoricals(row, encoders=_encoders, fit=False)
        row, _ = scale_features(row, scaler=_scaler, fit=False)

        for col in FEATURE_COLS:
            if col not in row.columns:
                row[col] = 0
        row = row[FEATURE_COLS]

        predicted = float(_model.predict(row)[0])
        predicted = round(predicted, 2)

        return JSONResponse({
            "price_lakhs": predicted,
            "price_crore": round(predicted / 100, 3) if predicted >= 100 else None,
            "price_formatted": f"₹ {predicted:.2f} Lakhs",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/charts")
async def list_charts():
    """List all available chart images."""
    charts = []
    images_dir = Path("images")
    if images_dir.exists():
        for f in sorted(images_dir.glob("*.png")):
            charts.append({"filename": f.name, "url": f"/images/{f.name}"})
    return JSONResponse({"charts": charts})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  House Price Prediction - Web Dashboard")
    print("  Open: http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
