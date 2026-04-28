# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
main.py
-------
Orchestrator script -- runs the full pipeline end-to-end:

    Step 1 → Generate / load dataset
    Step 2 → Train all models
    Step 3 → Evaluate best model
    Step 4 → Generate all visualisation plots
    Step 5 → Demo prediction

Usage:
    python main.py
    python main.py --predict-only   (run prediction only, skip training)
"""

import os
import sys
import argparse

# ── ensure src/ is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train      import train
from src.evaluate   import evaluate_model
from src.visualize  import generate_all_plots
from src.predict    import predict

DEMO_PROPERTY = {
    "area_sqft":        1200,
    "bedrooms":         3,
    "bathrooms":        2,
    "balconies":        1,
    "location":         "Andheri",
    "age_years":        5,
    "parking":          1,
    "furnishing":       "Semi-Furnished",
    "floor":            7,
    "total_floors":     15,
    "distance_city_km": 8.5,
    "price_lakhs":      0,
}


def banner(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_full_pipeline():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images",  exist_ok=True)

    banner("STEP 1 & 2 — Data Generation + Model Training")
    train()

    banner("STEP 3 — Model Evaluation")
    evaluate_model()

    banner("STEP 4 — Visualisation")
    generate_all_plots()

    banner("STEP 5 — Demo Prediction (3BHK Andheri, 1200 sqft)")
    predict(DEMO_PROPERTY)

    print("\n\n✅ FULL PIPELINE COMPLETE!")
    print("   • Dataset  → data/housing_data.csv")
    print("   • Models   → models/")
    print("   • Outputs  → outputs/")
    print("   • Charts   → images/")


def run_predict_only():
    banner("HOUSE PRICE PREDICTION — Interactive Mode")
    predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="House Price Prediction using Regression Models"
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Run interactive prediction without re-training"
    )
    args = parser.parse_args()

    if args.predict_only:
        run_predict_only()
    else:
        run_full_pipeline()
