"""
data_generator.py
-----------------
Generates a realistic synthetic housing dataset that mimics
Indian real-estate market patterns.

Features generated:
    area_sqft, bedrooms, bathrooms, balconies, location,
    age_years, parking, furnishing, floor, total_floors,
    distance_city_km, price_lakhs (target)
"""

import numpy as np
import pandas as pd

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)


def generate_housing_data(n_samples: int = 2000) -> pd.DataFrame:
    """Return a synthetic housing DataFrame with *n_samples* rows."""

    # ── 1. Location & base price multiplier ─────────────────────────────────
    locations = {
        "Bandra":        4.5,
        "Andheri":       3.2,
        "Thane":         2.5,
        "Pune":          2.2,
        "Navi Mumbai":   2.8,
        "Borivali":      2.9,
        "Dadar":         3.8,
        "Worli":         5.0,
        "Powai":         3.5,
        "Nashik":        1.5,
    }
    location_names  = list(locations.keys())
    location_mults  = list(locations.values())
    loc_probs       = np.array([0.10, 0.15, 0.12, 0.12, 0.10,
                                0.10, 0.08, 0.07, 0.10, 0.06])

    loc_indices  = rng.choice(len(location_names), size=n_samples, p=loc_probs)
    location_col = [location_names[i] for i in loc_indices]
    loc_mult_col = np.array([location_mults[i] for i in loc_indices])

    # ── 2. Property characteristics ─────────────────────────────────────────
    area_sqft    = rng.integers(400, 4001, size=n_samples)
    bedrooms     = rng.choice([1, 2, 3, 4, 5],
                               size=n_samples, p=[0.10, 0.35, 0.35, 0.15, 0.05])
    bathrooms    = np.clip(bedrooms + rng.choice([-1, 0, 1], size=n_samples,
                                                  p=[0.1, 0.7, 0.2]), 1, 5)
    balconies    = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.2, 0.4, 0.3, 0.1])
    age_years    = rng.integers(0, 31, size=n_samples)
    parking      = rng.choice([0, 1, 2], size=n_samples, p=[0.3, 0.55, 0.15])
    furnishing   = rng.choice(
                       ["Unfurnished", "Semi-Furnished", "Furnished"],
                       size=n_samples, p=[0.35, 0.40, 0.25])

    total_floors = rng.integers(2, 31, size=n_samples)
    floor        = np.array([rng.integers(0, tf + 1) for tf in total_floors])

    distance_city_km = rng.uniform(0.5, 40, size=n_samples).round(1)

    # ── 3. Furnishing multiplier ─────────────────────────────────────────────
    furn_map = {"Unfurnished": 1.0, "Semi-Furnished": 1.08, "Furnished": 1.18}
    furn_mult = np.array([furn_map[f] for f in furnishing])

    # ── 4. Price formula ─────────────────────────────────────────────────────
    # Base: area-driven price
    base_price = (area_sqft * 3.5 * loc_mult_col * furn_mult)

    # Bedroom/bathroom bonus
    base_price += bedrooms * 2.0 * loc_mult_col * 50_000
    base_price += bathrooms * 1.5 * loc_mult_col * 30_000

    # Age penalty (newer = costlier)
    age_factor  = 1 - (age_years * 0.008)
    base_price *= np.clip(age_factor, 0.75, 1.0)

    # Parking bonus
    base_price += parking * loc_mult_col * 1_50_000

    # Higher floor bonus (above ground, capped effect)
    floor_ratio  = floor / np.maximum(total_floors, 1)
    base_price  *= (1 + floor_ratio * 0.05)

    # Distance penalty
    base_price  *= np.exp(-0.008 * distance_city_km)

    # Convert to lakhs + noise
    price_lakhs = (base_price / 1_00_000).round(2)
    noise       = rng.normal(loc=0, scale=price_lakhs * 0.05)
    price_lakhs = np.clip((price_lakhs + noise).round(2), 10, 30_000)

    # ── 5. Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "area_sqft":        area_sqft,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms.astype(int),
        "balconies":        balconies,
        "location":         location_col,
        "age_years":        age_years,
        "parking":          parking,
        "furnishing":       furnishing,
        "floor":            floor,
        "total_floors":     total_floors,
        "distance_city_km": distance_city_km,
        "price_lakhs":      price_lakhs,
    })

    # ── 6. Inject realistic missing values (2 %) ─────────────────────────────
    for col in ["balconies", "parking", "distance_city_km"]:
        mask = rng.random(n_samples) < 0.02
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_housing_data(2000)
    out = r"data/housing_data.csv"
    df.to_csv(out, index=False)
    print(f"✅ Dataset saved → {out}  |  shape: {df.shape}")
    print(df.describe())
